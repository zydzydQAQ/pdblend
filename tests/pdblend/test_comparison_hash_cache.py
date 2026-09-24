import hashlib
import os
import time

import pytest

from pdblend.bench.comparison_hash_cache import UnchangedFileHashes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_unchanged_receipt_bytes_are_read_once_but_final_clear_rechecks(tmp_path):
    path=tmp_path/'raw';path.write_bytes(b'actual measured bytes')
    calls=[];cache=UnchangedFileHashes(clock=lambda:time.time()+10.)
    def read(p):calls.append(p);return sha(p)
    assert cache.read(path,read)==sha(path)
    assert cache.read(path,read)==sha(path) and len(calls)==1
    cache.clear()
    assert cache.read(path,read)==sha(path) and len(calls)==2


@pytest.mark.parametrize('change',['content','restored_mtime','replace_inode','delete'])
def test_changed_or_removed_file_never_inherits_old_digest(tmp_path,change):
    path=tmp_path/'raw';path.write_bytes(b'old')
    cache=UnchangedFileHashes();old=cache.read(path,sha);stat=path.stat()
    if change=='delete':
        path.unlink()
        with pytest.raises(FileNotFoundError):cache.read(path,sha)
        return
    if change=='replace_inode':
        replacement=tmp_path/'replacement';replacement.write_bytes(b'new');replacement.replace(path)
    else:path.write_bytes(b'new')
    if change=='restored_mtime':os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    assert cache.read(path,sha)==sha(path)!=old


@pytest.mark.parametrize('change',['content','restored_mtime','replace_inode','delete'])
def test_previously_cached_historical_file_is_invalidated(tmp_path,change):
    path=tmp_path/'raw';path.write_bytes(b'old')
    # Treat the initial bytes as old so this test enters the real hit branch.
    cache=UnchangedFileHashes(clock=lambda:time.time()+10.);calls=[]
    def read(p):calls.append(p);return sha(p)
    old=cache.read(path,read);stat=path.stat()
    assert cache.entries and cache.read(path,read)==old and len(calls)==1
    # Ensure a later filesystem tick, independently of the cache's age clock.
    time.sleep(.025)
    if change=='delete':
        path.unlink()
        with pytest.raises(FileNotFoundError):cache.read(path,read)
        return
    if change=='replace_inode':
        replacement=tmp_path/'replacement';replacement.write_bytes(b'new');replacement.replace(path)
    else:path.write_bytes(b'new')
    if change=='restored_mtime':os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    assert cache.read(path,read)==sha(path)!=old and len(calls)==2


def test_mid_read_mutation_is_rejected_and_does_not_poison_cache(tmp_path):
    path=tmp_path/'raw';path.write_bytes(b'old')
    cache=UnchangedFileHashes()
    def unstable(p):
        value=sha(p);p.write_bytes(b'changed during hash');return value
    with pytest.raises(ValueError,match='changed while hashing'):cache.read(path,unstable)
    assert not cache.entries
    assert cache.read(path,sha)==sha(path)


def test_cache_bounds_memory_and_evicts_the_least_recent_file(tmp_path):
    files=[tmp_path/str(i) for i in range(3)]
    for i,p in enumerate(files):p.write_text(str(i))
    cache=UnchangedFileHashes(capacity=2,clock=lambda:time.time()+10.);calls=[]
    def read(p):calls.append(p);return sha(p)
    for p in (files[0],files[1],files[0],files[2]):cache.read(p,read)
    assert len(cache.entries)==2 and len(calls)==3
    cache.read(files[1],read)
    assert len(calls)==4


def test_different_path_aliases_share_only_verified_actual_file_bytes(tmp_path):
    path=tmp_path/'raw';path.write_bytes(b'one immutable artifact')
    alias=tmp_path/'link';alias.symlink_to(path)
    cache=UnchangedFileHashes(clock=lambda:time.time()+10.);calls=[]
    def read(p):calls.append(p);return sha(p)
    assert cache.read(path,read)==cache.read(alias,read) and len(calls)==1


def test_export_hash_wrapper_is_opt_in_and_still_returns_actual_digest(tmp_path,monkeypatch):
    from pdblend.bench import comparison_campaign as campaign
    path=tmp_path/'raw';path.write_bytes(b'data');calls=[]
    def read(p):calls.append(p);return sha(p)
    monkeypatch.setattr(campaign,'_hash_file_bytes',read)
    monkeypatch.setattr(campaign,'_WATCH_DIGEST_CACHE',None)
    campaign.file_sha(path);campaign.file_sha(path)
    assert len(calls)==2
    monkeypatch.setattr(campaign,'_WATCH_DIGEST_CACHE',UnchangedFileHashes(clock=lambda:time.time()+10.))
    campaign.file_sha(path);campaign.file_sha(path)
    assert len(calls)==3


def test_coarse_timestamp_recent_writes_are_always_rehashed(tmp_path,monkeypatch):
    path=tmp_path/'raw';path.write_bytes(b'old')
    cache=UnchangedFileHashes();fixed=cache.identity(path)
    monkeypatch.setattr(cache,'identity',lambda p:fixed)
    old=cache.read(path,sha);path.write_bytes(b'new')
    assert cache.read(path,sha)==sha(path)!=old
    assert not cache.entries
