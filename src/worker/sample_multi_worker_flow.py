import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from resume_lib import (
    claim_next_row,
    complete_row,
    release_worker_claims,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simulate the resumable multi-worker extraction flow.'
    )
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rows', type=int, default=20)
    parser.add_argument('--sleep_seconds', type=float, default=0.05)
    parser.add_argument('--keep_temp', action='store_true')

    return parser.parse_args()


def write_jsonl(path, rows):
    with open(path, 'w') as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + '\n')


def read_jsonl(path):
    with open(path) as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def build_dataset(row_count):
    rows = []
    for index in range(row_count):
        rows.append(
            {
                'index': index,
                'example_id': index // 5,
                'utterance_id': index % 5,
                'type': ['Sample'],
                'modality': 'text',
                'utterance': f'sample utterance {index}',
                'processed': False,
                'read': False,
            }
        )

    return rows


def worker_process(worker_id, dataset_path, output_path, sleep_seconds):
    lock_path = None
    try:
        release_worker_claims(str(dataset_path), str(worker_id), lock_path)
        while True:
            row = claim_next_row(str(dataset_path), str(worker_id), lock_path)
            if row is None:
                return 0

            time.sleep(sleep_seconds)
            extraction = {
                'fake_model': True,
                'worker_id': worker_id,
                'text': row['utterance'],
            }
            complete_row(
                str(dataset_path), str(output_path), row['index'], extraction, lock_path
            )
    except Exception as exc:
        print(f'Worker {worker_id} error: {exc}', file=sys.stderr)
        return 1


def start_workers(work_dir, worker_count, sleep_seconds):
    dataset_path = work_dir / 'dataset.jsonl'
    output_path = work_dir / 'outputs.jsonl'

    workers = []
    for worker_id in range(worker_count):
        proc = subprocess.Popen(
            [
                sys.executable,
                '-c',
                f"""
import sys
sys.path.insert(0, '{Path(__file__).parent}')
from sample_multi_worker_flow import worker_process
sys.exit(worker_process({worker_id}, {repr(str(dataset_path))}, {repr(str(output_path))}, {sleep_seconds}))
""",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        workers.append(proc)

    return workers


def wait_for_workers(workers):
    failures = []
    for worker_id, worker in enumerate(workers):
        stdout, stderr = worker.communicate()
        if worker.returncode != 0:
            failures.append(
                {
                    'worker_id': worker_id,
                    'returncode': worker.returncode,
                    'stdout': stdout,
                    'stderr': stderr,
                }
            )

    if failures:
        for failure in failures:
            print(
                f'Worker {failure["worker_id"]} failed with '
                f'exit code {failure["returncode"]}',
                file=sys.stderr,
            )
            print(failure['stdout'], file=sys.stderr)
            print(failure['stderr'], file=sys.stderr)
        raise RuntimeError('One or more workers failed.')


def validate_results(dataset_path, output_path, expected_rows):
    dataset_rows = read_jsonl(dataset_path)
    output_rows = read_jsonl(output_path)
    output_indexes = [row['index'] for row in output_rows]

    if len(dataset_rows) != expected_rows:
        raise AssertionError(f'Expected {expected_rows} dataset rows.')
    if len(output_rows) != expected_rows:
        raise AssertionError(f'Expected {expected_rows} output rows.')
    if sorted(output_indexes) != list(range(expected_rows)):
        raise AssertionError('Output indexes are missing or duplicated.')

    for row in dataset_rows:
        if row.get('read') is not True or row.get('processed') is not True:
            raise AssertionError(f'Row {row.get("index")} was not completed.')

    for row in output_rows:
        if 'read' in row or 'processed' in row:
            raise AssertionError('Output rows should not contain state flags.')
        if row.get('extraction', {}).get('fake_model') is not True:
            raise AssertionError(f'Output row {row.get("index")} is malformed.')


def main():
    args = parse_args()
    work_dir = Path(tempfile.mkdtemp(prefix='multi-worker-flow-', dir='/private/tmp'))

    try:
        dataset_path = work_dir / 'dataset.jsonl'
        output_path = work_dir / 'outputs.jsonl'

        write_jsonl(dataset_path, build_dataset(args.rows))

        started_at = time.time()
        workers = start_workers(
            work_dir=work_dir,
            worker_count=args.workers,
            sleep_seconds=args.sleep_seconds,
        )
        wait_for_workers(workers)
        validate_results(dataset_path, output_path, args.rows)

        elapsed = time.time() - started_at
        print(
            f'PASS: {args.workers} workers processed {args.rows} rows '
            f'without duplicates in {elapsed:.2f}s.'
        )
        if args.keep_temp:
            print(f'Temp dataset: {dataset_path}')
            print(f'Temp output:  {output_path}')
    finally:
        if args.keep_temp:
            print(f'Kept temp directory: {work_dir}')
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
