import argparse
import json
from pathlib import Path

import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Prepare JSONL variants or a text file containing one utterance per line.'
        )
    )
    parser.add_argument(
        '--input_path', default='datasets/classification_dataset_claude_5_tones.jsonl'
    )
    parser.add_argument(
        '--output_path', default='datasets/preprocessed_dataset_claude_5_tones.jsonl'
    )
    parser.add_argument(
        '--input_format',
        choices=('auto', 'jsonl', 'txt'),
        default='auto',
        help='Input format. By default, .txt files are treated as plain text.',
    )

    return parser.parse_args()


def preprocess_jsonl(input_path, output_path):
    index = 0

    with open(input_path) as input_file, open(output_path, 'w') as output_file:
        for example_id, line in tqdm.tqdm(
            enumerate(input_file),
            desc='Expanding and preprocessing dataset',
        ):
            datapoint = json.loads(line)

            for utterance_id, variant in enumerate(datapoint['variants']):
                new_datapoint = {
                    'index': index,
                    'example_id': example_id,
                    'utterance_id': utterance_id,
                    'type': datapoint['type'],
                    'modality': datapoint['modality'],
                    'utterance': variant,
                    'processed': False,
                    'read': False,
                }
                output_file.write(json.dumps(new_datapoint) + '\n')
                index += 1
    return index


def preprocess_txt(input_path, output_path):
    index = 0

    with open(input_path) as input_file, open(output_path, 'w') as output_file:
        for line in tqdm.tqdm(
            input_file,
            desc='Preprocessing utterances',
        ):
            utterance = line.strip()
            if not utterance:
                continue

            new_datapoint = {
                'index': index,
                'example_id': index,
                'utterance_id': 0,
                'type': [],
                'modality': 'text',
                'utterance': utterance,
                'processed': False,
                'read': False,
            }
            output_file.write(json.dumps(new_datapoint) + '\n')
            index += 1

    return index


def resolve_input_format(input_path, input_format):
    if input_format != 'auto':
        return input_format

    return 'txt' if Path(input_path).suffix.lower() == '.txt' else 'jsonl'


def main():
    args = parse_args()
    input_format = resolve_input_format(args.input_path, args.input_format)
    if input_format == 'txt':
        count = preprocess_txt(args.input_path, args.output_path)
    else:
        count = preprocess_jsonl(args.input_path, args.output_path)

    print(f'Successfully processed {count} utterances.')


if __name__ == '__main__':
    main()
