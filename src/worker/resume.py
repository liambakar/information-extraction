import argparse
import sys

from resume_lib import (
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_PATH,
    release_worker_claims,
    reset_all_rows,
    reset_incomplete_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Manage resumable JSONL extraction.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    reset_parser = subparsers.add_parser(
        'reset-all',
        help='Reset all incomplete rows (read=True, processed=False) back to read=False.',
    )
    add_dataset_args(reset_parser)

    reset_scratch_parser = subparsers.add_parser(
        'reset-scratch',
        help=(
            'Full wipe: reset EVERY row (read=False, processed=False, clears '
            'claimed_by) and delete the output file. Use to restart a run from '
            'scratch. This is more destructive than reset-all.'
        ),
    )
    add_dataset_args(reset_scratch_parser)
    reset_scratch_parser.add_argument('--output_path', default=DEFAULT_OUTPUT_PATH)

    return parser.parse_args()


def add_dataset_args(parser):
    parser.add_argument('--dataset_path', default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        '--lock_path',
        default=None,
        help='Path to the shared lock file. Defaults to <dataset_path>.lock.',
    )


def run_reset_all(args):
    count = reset_incomplete_rows(
        dataset_path=args.dataset_path,
        lock_path=args.lock_path,
    )
    print(f'Reset {count} incomplete rows')
    return 0


def run_reset_scratch(args):
    count = reset_all_rows(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        lock_path=args.lock_path,
    )
    print(f'Reset {count} rows and cleared output at {args.output_path}')
    return 0


def main():
    args = parse_args()
    command_handlers = {
        'reset-all': run_reset_all,
        'reset-scratch': run_reset_scratch,
    }

    try:
        return command_handlers[args.command](args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
