"""Build immutable eight-GPU resident jobs for the existing lease worker."""
from pathlib import Path

from .resident_session import digest


def resident_job(group, group_path, *, root, source, image, verification, campaign, priority):
    root, source = Path(root).resolve(), Path(source).resolve()
    source_sha = source.name
    size = group['model_id'].split('-')[1].lower()
    job_id = 'comparison-' + size + '-' + digest(group)[:16]
    argv = ['docker', 'run', '--rm', '--name', job_id, '--gpus', 'all',
            '--cap-add', 'SYS_ADMIN', '--ipc=host', '--network=host', '--shm-size=16g',
            '--ulimit', 'nofile=65536:65536', '--entrypoint', '/opt/venv/bin/python']
    for src, dst, mode in [(str(source), '/opt/pdblend-src', 'ro'),
                           (str(root), str(root), 'ro'), ('/home/models', '/models', 'ro'),
                           ('/tmp/pdblend-physical-clock-owners', '/tmp/pdblend-physical-clock-owners', 'rw'),
                           ('{attempt_dir}', '{attempt_dir}', 'rw'), ('{attempt_dir}', '/output', 'rw')]:
        argv += ['-v', f'{src}:{dst}:{mode}']
    env = dict(group['engine_identity']['environment'], PYTHONPATH='/opt/pdblend-src',
        PYTHONDONTWRITEBYTECODE='1', PDBLEND_SOURCE_MANIFEST=str(source/'manifest.json'),
        PDBLEND_SOURCE_SHA256=source_sha, PDBLEND_IMAGE_ID=image, PDBLEND_MODELS_DIR='/models',
        PDBLEND_MODEL_VERIFICATION_RECEIPT=str(Path(verification).resolve()),
        PDBLEND_GPU_UUIDS='{lease_gpu_uuids}',
        PDBLEND_CLOCK_LOCK_DIR='/tmp/pdblend-physical-clock-owners',
        PDBLEND_CONCURRENCY_ENVIRONMENT='/output/concurrency-environment.json')
    for key, value in env.items():
        argv += ['-e', key+'='+value]
    argv += [image, '-B', '-m', 'pdblend.bench.comparison_runtime', '--group', str(Path(group_path).resolve()),
             '--out', '{attempt_dir}/session', '--base-port', '{lease_port}']
    return dict(job_id=job_id, priority=priority, max_attempts=1, payload=dict(
        argv=argv, container_name=job_id, gpu_count=8, exclusive=True, reserve_host=True,
        timeout_s=18000, required_receipts=['session/completion.json'], cwd=str(root),
        source_sha256=source_sha, image_digest=image, comparison_campaign=str(Path(campaign).resolve()),
        session_id=group['session_id']))
