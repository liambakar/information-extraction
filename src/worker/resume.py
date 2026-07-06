import argparse
import json
import sys

from resume_lib import (
    DEFAULT_CLAIM_TIMEOUT_SECONDS,
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_PATH,
    NO_WORK_EXIT_CODE,
    claim_next_row,
    complete_row,
    find_next_row,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Manage resumable JSONL extraction.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    find_next_parser = subparsers.add_parser(
        'find-next',
        help='Print the first unread and unprocessed row without changing the dataset.',
    )
    add_dataset_args(find_next_parser)

    claim_next_parser = subparsers.add_parser(
        'claim-next',
        help='Claim the first unprocessed row and mark it as read.',
    )
    add_dataset_args(claim_next_parser)
    claim_next_parser.add_argument(
        '--claim_timeout_seconds',
        type=float,
        default=DEFAULT_CLAIM_TIMEOUT_SECONDS,
        help='Retry read-but-unprocessed rows only after this many seconds.',
    )

    complete_parser = subparsers.add_parser(
        'complete',
        help='Append extraction output and mark the source row as processed.',
    )
    add_dataset_args(complete_parser)
    complete_parser.add_argument('--output_path', default=DEFAULT_OUTPUT_PATH)
    complete_parser.add_argument('--index', type=int, required=True)
    complete_parser.add_argument('--extraction_path', required=True)

    return parser.parse_args()


def add_dataset_args(parser):
    parser.add_argument('--dataset_path', default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        '--lock_path',
        default=None,
        help='Path to the shared lock file. Defaults to <dataset_path>.lock.',
    )


def print_row_or_no_work(row):
    if row is None:
        return NO_WORK_EXIT_CODE

    print(json.dumps(row))
    return 0


def run_find_next(args):
    row = find_next_row(args.dataset_path)
    return print_row_or_no_work(row)


def run_claim_next(args):
    row = claim_next_row(
        dataset_path=args.dataset_path,
        lock_path=args.lock_path,
        claim_timeout_seconds=args.claim_timeout_seconds,
    )
    return print_row_or_no_work(row)


def run_complete(args):
    row = complete_row(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        index=args.index,
        extraction_path=args.extraction_path,
        lock_path=args.lock_path,
    )
    print(json.dumps(row))
    return 0


def main():
    args = parse_args()
    command_handlers = {
        'find-next': run_find_next,
        'claim-next': run_claim_next,
        'complete': run_complete,
    }

    try:
        return command_handlers[args.command](args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
