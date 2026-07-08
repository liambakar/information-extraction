import fcntl
import json
import os
import shutil
import time
from contextlib import contextmanager


def resolve_lock_path(dataset_path, lock_path):
    if lock_path:
        return lock_path

    return f'{dataset_path}.lock'


@contextmanager
def exclusive_lock(lock_path):
    directory = os.path.dirname(lock_path) or '.'
    os.makedirs(directory, exist_ok=True)
    cleanup_legacy_lock_directory(lock_path)

    with open(lock_path, 'a+') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        write_lock_owner(lock_file)
        try:
            yield
        finally:
            try:
                clear_lock_owner(lock_file)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def cleanup_legacy_lock_directory(lock_path):
    if not os.path.isdir(lock_path):
        return

    try:
        shutil.rmtree(lock_path)
    except FileNotFoundError:
        pass


def write_lock_owner(lock_file):
    owner = {
        'pid': os.getpid(),
        'created_at': time.time(),
    }

    lock_file.seek(0)
    lock_file.truncate()
    json.dump(owner, lock_file)
    lock_file.write('\n')
    lock_file.flush()
    os.fsync(lock_file.fileno())


def clear_lock_owner(lock_file):
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.flush()
    os.fsync(lock_file.fileno())
