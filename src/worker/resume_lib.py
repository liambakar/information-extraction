import json
import os
import shutil
import tempfile
from glob import glob

try:
    from .file_lock import exclusive_lock, resolve_lock_path
except ImportError:
    try:
        from src.worker.file_lock import exclusive_lock, resolve_lock_path
    except ImportError:
        from file_lock import exclusive_lock, resolve_lock_path


DEFAULT_DATASET_PATH = 'datasets/preprocessed_dataset_claude_5_tones.jsonl'
DEFAULT_OUTPUT_PATH = 'out/nuextract3_outputs.jsonl'
NO_WORK_EXIT_CODE = 2
OUTPUT_FIELDS = (
    'index',
    'example_id',
    'utterance_id',
    'type',
    'modality',
    'utterance',
)


def read_jsonl(path):
    rows = []
    with open(path) as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON on line {line_number}: {exc}') from exc

    return rows


def write_jsonl_atomic(path, rows):
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f'.{os.path.basename(path)}.',
        suffix='.tmp',
        text=True,
    )

    try:
        with os.fdopen(fd, 'w') as output_file:
            for row in rows:
                output_file.write(json.dumps(row) + '\n')
            output_file.flush()
            os.fsync(output_file.fileno())

        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def find_unread_unprocessed(rows):
    for row in rows:
        if row.get('processed') is False and row.get('read') is False:
            return row

    return None


def find_claimable(rows):
    for row in rows:
        if row.get('processed') is not False:
            continue
        if row.get('read') is not True:
            return row

    return None


def find_row_by_index(rows, index):
    for row in rows:
        if row.get('index') == index:
            return row

    return None


def cleanup_stale_locks(dataset_path):
    directory = os.path.dirname(dataset_path) or '.'
    lock_pattern = os.path.join(directory, '*.lock')
    for lock_dir in glob(lock_pattern):
        try:
            if os.path.isdir(lock_dir):
                shutil.rmtree(lock_dir)
        except Exception:
            pass


def release_worker_claims(dataset_path, worker_id, lock_path=None):
    lock_path = resolve_lock_path(dataset_path, lock_path)

    with exclusive_lock(lock_path):
        rows = read_jsonl(dataset_path)
        count = 0
        for row in rows:
            if (
                row.get('claimed_by') == worker_id
                and row.get('read') is True
                and row.get('processed') is not True
            ):
                row['read'] = False
                row.pop('claimed_by', None)
                row.pop('claimed_at', None)
                count += 1
        write_jsonl_atomic(dataset_path, rows)
        return count


def reset_incomplete_rows(dataset_path, lock_path=None):
    cleanup_stale_locks(dataset_path)
    rows = read_jsonl(dataset_path)
    count = 0
    for row in rows:
        if row.get('read') is True and row.get('processed') is not True:
            row['read'] = False
            row.pop('claimed_by', None)
            row.pop('claimed_at', None)
            count += 1
    write_jsonl_atomic(dataset_path, rows)
    return count


def reset_all_rows(dataset_path, output_path, lock_path=None):
    cleanup_stale_locks(dataset_path)
    rows = read_jsonl(dataset_path)
    for row in rows:
        row['read'] = False
        row['processed'] = False
        row.pop('claimed_by', None)
        row.pop('claimed_at', None)
    write_jsonl_atomic(dataset_path, rows)
    if output_path and os.path.exists(output_path):
        os.remove(output_path)
    return len(rows)


def find_next_row(dataset_path):
    rows = read_jsonl(dataset_path)
    return find_unread_unprocessed(rows)


def claim_next_row(dataset_path, worker_id, lock_path=None):
    lock_path = resolve_lock_path(dataset_path, lock_path)

    with exclusive_lock(lock_path):
        rows = read_jsonl(dataset_path)
        row = find_claimable(rows)
        if row is None:
            return None

        row['claimed_by'] = worker_id
        row['read'] = True
        write_jsonl_atomic(dataset_path, rows)
        return row


def read_extraction(path):
    with open(path) as extraction_file:
        extraction_text = extraction_file.read().strip()

    try:
        return json.loads(extraction_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'Extraction output is not valid JSON: {exc}') from exc


def output_contains_index(output_path, index):
    if not os.path.exists(output_path):
        return False

    with open(output_path) as output_file:
        for line_number, line in enumerate(output_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'Invalid JSON in output file on line {line_number}: {exc}'
                ) from exc

            if row.get('index') == index:
                return True

    return False


def append_output(output_path, source_row, extraction):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    output_row = {field: source_row.get(field) for field in OUTPUT_FIELDS}
    output_row['extraction'] = extraction

    with open(output_path, 'a') as output_file:
        output_file.write(json.dumps(output_row) + '\n')
        output_file.flush()
        os.fsync(output_file.fileno())


def complete_row(dataset_path, output_path, index, extraction, lock_path=None):
    lock_path = resolve_lock_path(dataset_path, lock_path)

    with exclusive_lock(lock_path):
        rows = read_jsonl(dataset_path)
        source_row = find_row_by_index(rows, index)

        if source_row is None:
            raise ValueError(f'No row found with index {index}')

        if not output_contains_index(output_path, index):
            append_output(output_path, source_row, extraction)

        source_row['read'] = True
        source_row['processed'] = True
        source_row.pop('claimed_by', None)
        write_jsonl_atomic(dataset_path, rows)
        return source_row
