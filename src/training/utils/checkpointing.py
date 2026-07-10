import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


_MAX_SAFE_RUN_NAME_LENGTH = 80
_MISSING_RUN_NAME_CHECKPOINT_ID = 'unnamed-run'


@dataclass(frozen=True)
class CheckpointResolution:
    checkpoint_dir: Path
    resume_path: str | None
    message: str


def make_run_checkpoint_id(run_name: str) -> str:
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '-', run_name.strip())
    safe_name = safe_name.strip('.-_')[:_MAX_SAFE_RUN_NAME_LENGTH]

    if not safe_name:
        safe_name = 'run'

    run_hash = hashlib.sha1(run_name.encode('utf-8')).hexdigest()[:8]
    return f'{safe_name}-{run_hash}'


def get_run_checkpoint_dir(output_dir: str | Path, run_name: str | None) -> Path:
    checkpoint_id = (
        make_run_checkpoint_id(run_name)
        if run_name
        else _MISSING_RUN_NAME_CHECKPOINT_ID
    )
    return Path(output_dir) / 'checkpoints' / checkpoint_id


def resolve_checkpointing(
    output_dir: str | Path,
    run_name: str | None,
    resume_from: str | None,
) -> CheckpointResolution:
    checkpoint_dir = get_run_checkpoint_dir(output_dir, run_name)

    if resume_from is None:
        return CheckpointResolution(
            checkpoint_dir=checkpoint_dir,
            resume_path=None,
            message='No resume checkpoint requested.',
        )

    if resume_from != 'last':
        return CheckpointResolution(
            checkpoint_dir=checkpoint_dir,
            resume_path=resume_from,
            message=f'Using explicit resume checkpoint: {resume_from}',
        )

    if not run_name:
        return CheckpointResolution(
            checkpoint_dir=checkpoint_dir,
            resume_path=None,
            message=(
                'Requested resume_from="last", but no run name is set; '
                'starting fresh because the last checkpoint cannot be '
                'validated against the current run.'
            ),
        )

    last_checkpoint = checkpoint_dir / 'last.ckpt'
    if last_checkpoint.exists():
        return CheckpointResolution(
            checkpoint_dir=checkpoint_dir,
            resume_path=str(last_checkpoint),
            message=f'Resuming from current run last checkpoint: {last_checkpoint}',
        )

    return CheckpointResolution(
        checkpoint_dir=checkpoint_dir,
        resume_path=None,
        message=(
            'Requested resume_from="last", but no last checkpoint exists for '
            f'run "{run_name}" at {last_checkpoint}; starting fresh.'
        ),
    )
