import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_DATASET_PATH = REPO_ROOT / 'datasets' / 'preprocessed_mini_dataset.jsonl'
RESUME_SCRIPT_PATH = REPO_ROOT / 'src' / 'worker' / 'resume.py'
RESET_ENTRYPOINT_PATH = REPO_ROOT / 'scripts' / 'reset_entrypoint.sh'
SUBMIT_JOB_PATH = REPO_ROOT / 'scripts' / 'submit_extraction_job.sh'


def read_jsonl(path):
    with open(path) as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def install_fake_nuextract_dependencies():
    fake_torch = types.ModuleType('torch')
    fake_torch.bfloat16 = object()

    class FakeInferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_torch.inference_mode = lambda: FakeInferenceMode()
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fake_transformers = types.ModuleType('transformers')
    fake_transformers.AutoModelForImageTextToText = object()
    fake_transformers.AutoProcessor = object()

    sys.modules.setdefault('torch', fake_torch)
    sys.modules.setdefault('transformers', fake_transformers)


def import_worker_with_fake_dependencies():
    install_fake_nuextract_dependencies()
    return importlib.import_module('src.worker.worker')


def run_reset_all(dataset_path, lock_path):
    return subprocess.run(
        [
            sys.executable,
            str(RESUME_SCRIPT_PATH),
            'reset-all',
            '--dataset_path',
            str(dataset_path),
            '--lock_path',
            str(lock_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def build_helper_command(function_name, *args):
    arguments = ', '.join(repr(str(arg)) for arg in args)
    return [
        sys.executable,
        '-c',
        (
            'from tests.test_dummy_nuextract_workflow import '
            f'{function_name}; '
            f'raise SystemExit({function_name}({arguments}))'
        ),
    ]


def run_dummy_worker_process(worker_id, dataset_path, output_path, lock_path):
    worker = import_worker_with_fake_dependencies()
    dummy_processor = object()
    dummy_model = object()
    dummy_template = {'template': 'dummy'}

    def dummy_run_extraction(text, **kwargs):
        sleep_seconds = float(os.environ.get('DUMMY_WORKER_SLEEP_SECONDS', '0'))
        if sleep_seconds:
            time.sleep(sleep_seconds)
        return json.dumps({'dummy': True, 'text': text, 'worker_id': worker_id})

    environment = {
        'DATASET_PATH': str(dataset_path),
        'PROCESSED_DATA_PATH': str(output_path),
        'LOCK_PATH': str(lock_path),
        'SLURM_ARRAY_TASK_ID': str(worker_id),
        'MODEL_ID': 'dummy-nuextract',
    }

    with (
        patch.dict('os.environ', environment, clear=False),
        patch.object(
            worker,
            'load_model',
            Mock(return_value=(dummy_processor, dummy_model)),
        ),
        patch.object(worker, 'load_template', Mock(return_value=dummy_template)),
        patch.object(worker, 'run_extraction', Mock(side_effect=dummy_run_extraction)),
    ):
        worker.main()

    return 0


def run_crashing_worker_process(worker_id, dataset_path, output_path, lock_path):
    worker = import_worker_with_fake_dependencies()
    dummy_processor = object()
    dummy_model = object()
    dummy_template = {'template': 'dummy'}

    def crash_during_extraction(text, **kwargs):
        print(
            f'[worker {worker_id}] simulating crash while extracting {text!r}',
            flush=True,
        )
        os._exit(17)

    environment = {
        'DATASET_PATH': str(dataset_path),
        'PROCESSED_DATA_PATH': str(output_path),
        'LOCK_PATH': str(lock_path),
        'SLURM_ARRAY_TASK_ID': str(worker_id),
        'MODEL_ID': 'dummy-nuextract',
    }

    with (
        patch.dict('os.environ', environment, clear=False),
        patch.object(
            worker,
            'load_model',
            Mock(return_value=(dummy_processor, dummy_model)),
        ),
        patch.object(worker, 'load_template', Mock(return_value=dummy_template)),
        patch.object(
            worker,
            'run_extraction',
            Mock(side_effect=crash_during_extraction),
        ),
    ):
        worker.main()

    return 0


def collect_worker_outputs(workers, timeout=30):
    failures = []
    worker_outputs = []
    for worker_id, process in enumerate(workers):
        stdout, stderr = process.communicate(timeout=timeout)
        worker_outputs.append((worker_id, stdout, stderr))
        if process.returncode != 0:
            failures.append(
                (
                    worker_id,
                    process.returncode,
                    stdout,
                    stderr,
                )
            )

    return failures, worker_outputs


def print_worker_outputs(worker_outputs):
    for worker_id, stdout, stderr in worker_outputs:
        print(f'\n--- worker {worker_id} stdout ---')
        print(stdout, end='')
        if stderr:
            print(f'\n--- worker {worker_id} stderr ---')
            print(stderr, end='')


def wait_for_claimed_rows(dataset_path, minimum_count, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        claimed_rows = [
            row
            for row in read_jsonl(dataset_path)
            if (
                row.get('read') is True
                and row.get('processed') is not True
                and row.get('claimed_by')
            )
        ]
        if len(claimed_rows) >= minimum_count:
            return claimed_rows
        time.sleep(0.01)

    raise AssertionError(f'Timed out waiting for {minimum_count} claimed rows')


def wait_for_path(path, timeout=5):
    path = Path(path)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.01)

    raise AssertionError(f'Timed out waiting for {path}')


def hold_lock_until_released(lock_path, ready_path, release_path):
    from src.worker.file_lock import exclusive_lock

    with exclusive_lock(str(lock_path)):
        Path(ready_path).write_text('ready')
        deadline = time.time() + 10
        while not Path(release_path).exists() and time.time() < deadline:
            time.sleep(0.01)

    return 0


def acquire_lock_and_write(lock_path, acquired_path):
    from src.worker.file_lock import exclusive_lock

    with exclusive_lock(str(lock_path)):
        Path(acquired_path).write_text('acquired')

    return 0


def acquire_lock_and_die(lock_path, ready_path):
    from src.worker.file_lock import exclusive_lock

    with exclusive_lock(str(lock_path)):
        Path(ready_path).write_text('ready')
        os._exit(23)

    return 0


class DummyNuExtractWorkflowTest(unittest.TestCase):
    def test_file_lock_blocks_other_processes_until_released(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            lock_path = temp_dir_path / 'dataset.lock'
            ready_path = temp_dir_path / 'ready'
            release_path = temp_dir_path / 'release'
            acquired_path = temp_dir_path / 'acquired'
            holder_stdout = ''
            holder_stderr = ''
            contender_stdout = ''
            contender_stderr = ''
            contender = None

            holder = subprocess.Popen(
                build_helper_command(
                    'hold_lock_until_released',
                    lock_path,
                    ready_path,
                    release_path,
                ),
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                wait_for_path(ready_path)
                contender = subprocess.Popen(
                    build_helper_command(
                        'acquire_lock_and_write',
                        lock_path,
                        acquired_path,
                    ),
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                time.sleep(0.2)
                self.assertFalse(acquired_path.exists())

                release_path.write_text('release')
                holder_stdout, holder_stderr = holder.communicate(timeout=5)
                contender_stdout, contender_stderr = contender.communicate(timeout=5)
            finally:
                release_path.write_text('release')
                if holder.poll() is None:
                    holder.terminate()
                    holder.communicate(timeout=5)
                if contender is not None and contender.poll() is None:
                    contender.terminate()
                    contender.communicate(timeout=5)

            self.assertEqual(
                holder.returncode,
                0,
                f'holder failed\nstdout:\n{holder_stdout}\nstderr:\n{holder_stderr}',
            )
            self.assertEqual(
                contender.returncode,
                0,
                (
                    'contender failed\n'
                    f'stdout:\n{contender_stdout}\nstderr:\n{contender_stderr}'
                ),
            )
            self.assertTrue(acquired_path.exists())
            self.assertTrue(lock_path.is_file())

    def test_file_lock_releases_when_process_dies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            lock_path = temp_dir_path / 'dataset.lock'
            ready_path = temp_dir_path / 'ready'
            acquired_path = temp_dir_path / 'acquired'

            process = subprocess.Popen(
                build_helper_command('acquire_lock_and_die', lock_path, ready_path),
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            wait_for_path(ready_path)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(
                process.returncode,
                23,
                f'crash process failed unexpectedly\nstdout:\n{stdout}\nstderr:\n{stderr}',
            )

            result = subprocess.run(
                build_helper_command('acquire_lock_and_write', lock_path, acquired_path),
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                (
                    'lock was not reacquired after process death\n'
                    f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
                ),
            )
            self.assertTrue(acquired_path.exists())
            self.assertTrue(lock_path.is_file())

    def test_file_lock_converts_legacy_lock_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            lock_path = temp_dir_path / 'dataset.lock'
            acquired_path = temp_dir_path / 'acquired'
            lock_path.mkdir()
            (lock_path / 'owner.json').write_text('{}')

            result = subprocess.run(
                build_helper_command('acquire_lock_and_write', lock_path, acquired_path),
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                (
                    'legacy lock directory was not converted\n'
                    f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
                ),
            )
            self.assertTrue(acquired_path.exists())
            self.assertTrue(lock_path.is_file())

    def test_submit_job_generated_script_preserves_batch_exit_code(self):
        script = SUBMIT_JOB_PATH.read_text()

        self.assertNotIn('\nexit 0\n', script)
        self.assertIn('exec ./scripts/batch_extraction.sh', script)

    def test_submit_job_requires_dataset_path_argument(self):
        result = subprocess.run(
            [
                'bash',
                str(SUBMIT_JOB_PATH),
                'nuextract3_run',
                'out/slurm_logs',
                '8',
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('DATASET_PATH', result.stderr)

    def test_submit_job_requires_processed_data_path_argument(self):
        result = subprocess.run(
            [
                'bash',
                str(SUBMIT_JOB_PATH),
                'nuextract3_run',
                'out/slurm_logs',
                '8',
                'datasets/preprocessed_dataset_claude_5_tones.jsonl',
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('PROCESSED_DATA_PATH', result.stderr)

    def test_reset_entrypoint_requires_dataset_path_argument(self):
        result = subprocess.run(
            ['bash', str(RESET_ENTRYPOINT_PATH)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('DATASET_PATH', result.stderr)

    def test_worker_fails_fast_without_dataset_path(self):
        worker = import_worker_with_fake_dependencies()

        with patch.dict(
            'os.environ',
            {'SLURM_ARRAY_TASK_ID': 'worker_1'},
            clear=True,
        ):
            result = worker.main()

        self.assertEqual(result, 1)

    def test_worker_fails_fast_without_processed_data_path(self):
        worker = import_worker_with_fake_dependencies()

        with patch.dict(
            'os.environ',
            {
                'DATASET_PATH': 'datasets/preprocessed_dataset_claude_5_tones.jsonl',
                'SLURM_ARRAY_TASK_ID': 'worker_1',
            },
            clear=True,
        ):
            result = worker.main()

        self.assertEqual(result, 1)

    def test_reset_all_then_worker_processes_mini_dataset_with_dummy_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            dataset_path = temp_dir_path / 'mini_dataset.jsonl'
            output_path = temp_dir_path / 'dummy_outputs.jsonl'
            lock_path = temp_dir_path / 'mini_dataset.lock'
            shutil.copyfile(MINI_DATASET_PATH, dataset_path)

            original_rows = read_jsonl(dataset_path)
            stuck_rows = [
                row
                for row in original_rows
                if row.get('read') is True and row.get('processed') is not True
            ]
            self.assertTrue(
                stuck_rows,
                'mini_dataset.jsonl should include at least one incomplete claimed row',
            )
            self.assertTrue(
                any(row.get('claimed_by') == 'worker_1' for row in stuck_rows),
                'mini_dataset.jsonl should include a worker_1 claim to reset',
            )

            result = run_reset_all(dataset_path, lock_path)

            self.assertEqual(
                result.returncode,
                0,
                f'reset-all failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}',
            )

            reset_rows = read_jsonl(dataset_path)
            for row in reset_rows:
                if row.get('processed') is not True:
                    self.assertIs(row.get('read'), False)
                    self.assertNotIn('claimed_by', row)

            worker = import_worker_with_fake_dependencies()
            dummy_processor = object()
            dummy_model = object()
            dummy_template = {'template': 'dummy'}
            extraction_calls = []

            def dummy_run_extraction(text, **kwargs):
                extraction_calls.append({'text': text, **kwargs})
                return json.dumps({'dummy': True, 'text': text})

            environment = {
                'DATASET_PATH': str(dataset_path),
                'PROCESSED_DATA_PATH': str(output_path),
                'LOCK_PATH': str(lock_path),
                'SLURM_ARRAY_TASK_ID': 'worker_1',
                'MODEL_ID': 'dummy-nuextract',
            }

            with (
                patch.dict('os.environ', environment, clear=False),
                patch.object(
                    worker,
                    'load_model',
                    Mock(return_value=(dummy_processor, dummy_model)),
                ) as load_model_mock,
                patch.object(
                    worker,
                    'load_template',
                    Mock(return_value=dummy_template),
                ) as load_template_mock,
                patch.object(
                    worker,
                    'run_extraction',
                    Mock(side_effect=dummy_run_extraction),
                ),
            ):
                worker.main()

            load_model_mock.assert_called_once_with('dummy-nuextract')
            load_template_mock.assert_called_once()
            self.assertEqual(len(extraction_calls), len(original_rows))

            output_rows = read_jsonl(output_path)
            self.assertEqual(len(output_rows), len(original_rows))

            expected_output_fields = {
                'index',
                'example_id',
                'utterance_id',
                'type',
                'modality',
                'utterance',
                'extraction',
            }
            for output_row in output_rows:
                self.assertEqual(set(output_row), expected_output_fields)
                self.assertEqual(
                    output_row['extraction'],
                    {'dummy': True, 'text': output_row['utterance']},
                )

            final_rows = read_jsonl(dataset_path)
            for row in final_rows:
                self.assertIs(row.get('processed'), True)
                self.assertIs(row.get('read'), True)
                self.assertNotIn('claimed_by', row)

    def test_multiple_workers_process_mini_dataset_once_each(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            dataset_path = temp_dir_path / 'mini_dataset.jsonl'
            output_path = temp_dir_path / 'dummy_outputs.jsonl'
            lock_path = temp_dir_path / 'mini_dataset.lock'
            shutil.copyfile(MINI_DATASET_PATH, dataset_path)

            original_rows = read_jsonl(dataset_path)
            result = run_reset_all(dataset_path, lock_path)
            self.assertEqual(
                result.returncode,
                0,
                f'reset-all failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}',
            )

            workers = []
            for worker_id in range(4):
                command = build_helper_command(
                    'run_dummy_worker_process',
                    worker_id,
                    dataset_path,
                    output_path,
                    lock_path,
                )
                env = os.environ.copy()
                env['DUMMY_WORKER_SLEEP_SECONDS'] = '0.02'
                workers.append(
                    subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )

            failures, worker_outputs = collect_worker_outputs(workers)
            self.assertEqual(failures, [])
            print_worker_outputs(worker_outputs)

            output_rows = read_jsonl(output_path)
            output_indexes = [row['index'] for row in output_rows]
            self.assertEqual(len(output_rows), len(original_rows))
            self.assertEqual(
                sorted(output_indexes),
                sorted(row['index'] for row in original_rows),
            )
            self.assertEqual(len(output_indexes), len(set(output_indexes)))

            worker_ids = {row['extraction']['worker_id'] for row in output_rows}
            self.assertGreater(len(worker_ids), 1)

            for output_row in output_rows:
                self.assertEqual(
                    output_row['extraction'],
                    {
                        'dummy': True,
                        'text': output_row['utterance'],
                        'worker_id': output_row['extraction']['worker_id'],
                    },
                )

            final_rows = read_jsonl(dataset_path)
            for row in final_rows:
                self.assertIs(row.get('processed'), True)
                self.assertIs(row.get('read'), True)
                self.assertNotIn('claimed_by', row)

    def test_reset_all_recovers_claim_after_worker_crash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            dataset_path = temp_dir_path / 'mini_dataset.jsonl'
            output_path = temp_dir_path / 'dummy_outputs.jsonl'
            lock_path = temp_dir_path / 'mini_dataset.lock'
            shutil.copyfile(MINI_DATASET_PATH, dataset_path)

            original_rows = read_jsonl(dataset_path)
            result = run_reset_all(dataset_path, lock_path)
            self.assertEqual(
                result.returncode,
                0,
                f'reset-all failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}',
            )

            command = build_helper_command(
                'run_crashing_worker_process',
                'crash_worker',
                dataset_path,
                output_path,
                lock_path,
            )
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(timeout=10)
            print('\n--- crashing worker stdout ---')
            print(stdout, end='')
            if stderr:
                print('\n--- crashing worker stderr ---')
                print(stderr, end='')

            self.assertEqual(process.returncode, 17)
            rows_after_crash = read_jsonl(dataset_path)
            stuck_rows = [
                row
                for row in rows_after_crash
                if (
                    row.get('claimed_by') == 'crash_worker'
                    and row.get('read') is True
                    and row.get('processed') is not True
                )
            ]
            self.assertEqual(len(stuck_rows), 1)
            self.assertFalse(output_path.exists())

            result = run_reset_all(dataset_path, lock_path)
            self.assertEqual(
                result.returncode,
                0,
                f'reset-all failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}',
            )
            print(f'\n--- reset after crash stdout ---\n{result.stdout}', end='')

            reset_rows = read_jsonl(dataset_path)
            recovered_row = next(
                row for row in reset_rows if row['index'] == stuck_rows[0]['index']
            )
            self.assertIs(recovered_row.get('read'), False)
            self.assertIs(recovered_row.get('processed'), False)
            self.assertNotIn('claimed_by', recovered_row)

            run_dummy_worker_process(
                'recovery_worker',
                dataset_path,
                output_path,
                lock_path,
            )
            output_rows = read_jsonl(output_path)
            self.assertEqual(len(output_rows), len(original_rows))
            self.assertEqual(
                len({row['index'] for row in output_rows}),
                len(original_rows),
            )

            final_rows = read_jsonl(dataset_path)
            for row in final_rows:
                self.assertIs(row.get('processed'), True)
                self.assertIs(row.get('read'), True)
                self.assertNotIn('claimed_by', row)

    def test_reset_all_during_active_workers_does_not_duplicate_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            dataset_path = temp_dir_path / 'mini_dataset.jsonl'
            output_path = temp_dir_path / 'dummy_outputs.jsonl'
            lock_path = temp_dir_path / 'mini_dataset.lock'
            shutil.copyfile(MINI_DATASET_PATH, dataset_path)

            original_rows = read_jsonl(dataset_path)
            result = run_reset_all(dataset_path, lock_path)
            self.assertEqual(
                result.returncode,
                0,
                f'reset-all failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}',
            )

            workers = []
            for worker_id in range(4):
                command = build_helper_command(
                    'run_dummy_worker_process',
                    f'mid_reset_{worker_id}',
                    dataset_path,
                    output_path,
                    lock_path,
                )
                env = os.environ.copy()
                env['DUMMY_WORKER_SLEEP_SECONDS'] = '0.10'
                workers.append(
                    subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )

            claimed_rows = wait_for_claimed_rows(dataset_path, minimum_count=4)
            print('\n--- claims before mid-run reset ---')
            for row in claimed_rows:
                print(f"[worker {row['claimed_by']}] claimed index={row['index']}")

            result = run_reset_all(dataset_path, lock_path)
            self.assertEqual(
                result.returncode,
                0,
                f'reset-all failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}',
            )
            print(f'\n--- mid-run reset stdout ---\n{result.stdout}', end='')

            failures, worker_outputs = collect_worker_outputs(workers)
            self.assertEqual(failures, [])
            print_worker_outputs(worker_outputs)

            output_rows = read_jsonl(output_path)
            output_indexes = [row['index'] for row in output_rows]
            self.assertEqual(len(output_rows), len(original_rows))
            self.assertEqual(len(output_indexes), len(set(output_indexes)))
            self.assertEqual(
                sorted(output_indexes),
                sorted(row['index'] for row in original_rows),
            )

            worker_ids = {row['extraction']['worker_id'] for row in output_rows}
            self.assertGreater(len(worker_ids), 1)

            final_rows = read_jsonl(dataset_path)
            for row in final_rows:
                self.assertIs(row.get('processed'), True)
                self.assertIs(row.get('read'), True)
                self.assertNotIn('claimed_by', row)


if __name__ == '__main__':
    unittest.main()
