import json
import os
import sys

from src.worker.resume_lib import (
    DEFAULT_OUTPUT_PATH,
    claim_next_row,
    complete_row,
    release_worker_claims,
)
from src.extraction.run_nuextract3 import load_model, load_template, run_extraction


def parse_bool_env(name, default=False):
    val = os.environ.get(name, '')
    if not val:
        return default
    return val.lower() in ('1', 'true', 'yes')


def determine_worker_id():
    return os.environ.get('SLURM_ARRAY_TASK_ID', 'local')


def log(worker_id, message):
    print(f'[worker {worker_id}] {message}', flush=True)


def log_error(worker_id, message):
    print(f'[worker {worker_id}] ERROR: {message}', file=sys.stderr, flush=True)


def main():
    worker_id = determine_worker_id()
    dataset_path = os.environ.get('DATASET_PATH')
    output_path = os.environ.get('OUTPUT_PATH', DEFAULT_OUTPUT_PATH)
    lock_path = os.environ.get('LOCK_PATH') or None
    template_path = os.environ.get(
        'TEMPLATE_PATH', 'extraction_templates/template.json'
    )
    model_id = os.environ.get('MODEL_ID', 'numind/NuExtract3')
    temperature = float(os.environ.get('TEMPERATURE', '0.2'))
    max_new_tokens = int(os.environ.get('MAX_NEW_TOKENS', '4096'))
    enable_thinking = parse_bool_env('ENABLE_THINKING', default=False)

    log(worker_id, 'starting up')
    if not dataset_path:
        log_error(worker_id, 'DATASET_PATH must be set by the job launcher.')
        return 1

    released = release_worker_claims(dataset_path, worker_id, lock_path)
    log(worker_id, f'released {released} stuck claim(s)')

    log(worker_id, f'loading model {model_id}')
    processor, model = load_model(model_id)
    template = load_template(template_path)
    log(worker_id, 'model and template loaded; entering claim loop')

    run_claim_loop(
        dataset_path,
        output_path,
        lock_path,
        worker_id,
        processor,
        model,
        template,
        temperature,
        max_new_tokens,
        enable_thinking,
    )
    return 0


def run_claim_loop(
    dataset_path,
    output_path,
    lock_path,
    worker_id,
    processor,
    model,
    template,
    temperature,
    max_new_tokens,
    enable_thinking,
):
    while True:
        row = claim_next_row(dataset_path, worker_id, lock_path)
        if row is None:
            log(worker_id, 'no claimable rows remain; exiting')
            return

        index = row['index']
        log(worker_id, f'claimed index={index}')
        try:
            extraction_text = run_extraction(
                text=row['utterance'],
                model=model,
                processor=processor,
                template=template,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
            )
            extraction = json.loads(extraction_text)
            complete_row(dataset_path, output_path, index, extraction, lock_path)
            log(worker_id, f'completed index={index}')
        except Exception as exc:
            log_error(worker_id, f'failed index={index}: {exc}')
            continue


if __name__ == '__main__':
    sys.exit(main())
