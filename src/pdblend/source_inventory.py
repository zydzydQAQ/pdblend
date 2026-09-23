"""Stable source inventory for newly prepared experiment artifacts.

The source root is located independently of an implementation module's folder.
A manifest includes both compatibility entrances and actual implementations.
Existing frozen manifests must never be rewritten to match changed source.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SOURCE_SUFFIXES = frozenset({'.py', '.json', '.md', '.yaml', '.yml', '.toml'})


def source_root() -> Path:
    return Path(__file__).resolve().parent.parent


def source_files(root: Path | str | None = None, *, packages=('pdblend', 'pdblend_baselines')) -> tuple[Path, ...]:
    root = Path(root) if root is not None else source_root()
    return tuple(sorted(path for package in packages for path in (root / package).rglob('*')
                        if path.is_file() and '__pycache__' not in path.parts
                        and (path.suffix in SOURCE_SUFFIXES or path.name == 'LICENSE')))


def implementation_hashes(root: Path | str | None = None, *, packages=('pdblend',)) -> dict[str, str]:
    """Bind every implementation in the selected package, including aliases."""
    root = Path(root) if root is not None else source_root()
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files(root, packages=packages)}


def bound_path(relative: str, root: Path | str | None = None) -> Path:
    """Resolve only source-relative manifest entries; reject path traversal."""
    root = (Path(root) if root is not None else source_root()).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'source inventory entry is unavailable: {relative}')
    return path


def verify_implementation(expected: dict[str, str], *, root: Path | str | None = None,
                          packages=('pdblend',)) -> None:
    if expected != implementation_hashes(root, packages=packages):
        raise ValueError('source implementation changed; prepare a new immutable artifact')
