import argparse
import json
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Expand JSONL datapoint variants into one utterance per line.'
    )
    parser.add_argument(
        '--input_path', default='datasets/classification_dataset_claude_5_tones.jsonl'
    )
    parser.add_argument(
        '--output_path', default='datasets/preprocessed_dataset_claude_5_tones.jsonl'
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
    print(f'Successfully processed {index + 1} lines.')


def main():
    args = parse_args()
    preprocess_jsonl(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
