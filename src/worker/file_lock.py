import json
import os
import time
from contextlib import contextmanager


LOCK_POLL_SECONDS = 0.1
LOCK_STALE_SECONDS = 60 * 60


def resolve_lock_path(dataset_path, lock_path):
    if lock_path:
        return lock_path

    return f'{dataset_path}.lock'


@contextmanager
def exclusive_lock(lock_path):
    acquire_lock(lock_path)
    try:
        yield
    finally:
        try:
            remove_lock(lock_path)
        except FileNotFoundError:
            pass


def acquire_lock(lock_path):
    directory = os.path.dirname(lock_path) or '.'
    os.makedirs(directory, exist_ok=True)

    while True:
        try:
            os.mkdir(lock_path)
            write_lock_owner(lock_path)
            return
        except FileExistsError:
            if lock_is_stale(lock_path):
                remove_lock(lock_path)
                continue

            time.sleep(LOCK_POLL_SECONDS)


def write_lock_owner(lock_path):
    owner_path = os.path.join(lock_path, 'owner.json')
    owner = {
        'pid': os.getpid(),
        'created_at': time.time(),
    }

    with open(owner_path, 'w') as owner_file:
        json.dump(owner, owner_file)


def lock_is_stale(lock_path):
    try:
        lock_age = time.time() - os.path.getmtime(lock_path)
    except FileNotFoundError:
        return False

    return lock_age > LOCK_STALE_SECONDS


def remove_lock(lock_path):
    if os.path.isdir(lock_path):
        owner_path = os.path.join(lock_path, 'owner.json')
        try:
            os.unlink(owner_path)
        except FileNotFoundError:
            pass
        os.rmdir(lock_path)
    else:
        os.unlink(lock_path)
