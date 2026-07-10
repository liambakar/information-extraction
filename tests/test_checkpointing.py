from pathlib import Path

from src.training.utils.checkpointing import (
    get_run_checkpoint_dir,
    resolve_checkpointing,
)


def test_last_checkpoint_resolves_for_same_run_name(tmp_path):
    checkpoint_dir = get_run_checkpoint_dir(tmp_path, 'experiment-a')
    checkpoint_dir.mkdir(parents=True)
    last_checkpoint = checkpoint_dir / 'last.ckpt'
    last_checkpoint.write_text('checkpoint')

    resolution = resolve_checkpointing(tmp_path, 'experiment-a', 'last')

    assert resolution.checkpoint_dir == checkpoint_dir
    assert resolution.resume_path == str(last_checkpoint)


def test_different_run_names_use_different_checkpoint_dirs(tmp_path):
    first_dir = get_run_checkpoint_dir(tmp_path, 'experiment-a')
    second_dir = get_run_checkpoint_dir(tmp_path, 'experiment-b')

    assert first_dir != second_dir
    assert first_dir.parent == second_dir.parent == tmp_path / 'checkpoints'


def test_last_checkpoint_without_matching_checkpoint_starts_fresh(tmp_path):
    resolution = resolve_checkpointing(tmp_path, 'experiment-a', 'last')

    assert resolution.resume_path is None
    assert 'starting fresh' in resolution.message


def test_last_checkpoint_without_run_name_starts_fresh(tmp_path):
    resolution = resolve_checkpointing(tmp_path, None, 'last')

    assert resolution.resume_path is None
    assert resolution.checkpoint_dir == Path(tmp_path) / 'checkpoints' / 'unnamed-run'
    assert 'no run name is set' in resolution.message


def test_explicit_checkpoint_path_is_preserved(tmp_path):
    checkpoint_path = '/tmp/manual.ckpt'

    resolution = resolve_checkpointing(tmp_path, 'experiment-a', checkpoint_path)

    assert resolution.resume_path == checkpoint_path
