"""Run a checkpoint on random dataset samples and write a Markdown report."""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.data.dataset import DATASET, END_SEQ, START_SEQ


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('dataset_type', choices=('full', 'partial'))
    parser.add_argument(
        '--template',
        type=Path,
        default=Path('extraction_templates/template.json'),
    )
    parser.add_argument('--output', type=Path)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--input', help='Custom utterance to run after the samples')
    parser.add_argument(
        '--category',
        choices=('activity', 'food', 'mood', 'symptom', 'treatment'),
        help='Template category for a custom input with a partial model',
    )
    parser.add_argument(
        '--ground-truth',
        help='Optional expected JSON or text for the custom input',
    )
    parser.add_argument(
        '--device',
        choices=('auto', 'cuda', 'mps', 'cpu'),
        default='auto',
    )
    return parser.parse_args()


def get_device(requested):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def markdown_block(text):
    text = text.strip()
    for suffix in ('<|im_end|>', '<|endoftext|>'):
        text = text.removesuffix(suffix).strip()

    language = 'text'
    try:
        text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        language = 'json'
    except json.JSONDecodeError:
        pass

    fence = '````' if '```' in text else '```'
    return f'{fence}{language}\n{text}\n{fence}'


def decode(tokenizer, token_ids):
    return tokenizer.decode(
        token_ids.tolist(),
        skip_special_tokens=False,
    ).strip()


def custom_prompt(input_text, template_path, dataset_type, category):
    with template_path.open(encoding='utf-8') as template_file:
        template = json.load(template_file)

    if dataset_type == 'partial':
        if category is None:
            raise ValueError('--category is required for partial custom input.')
        template = template[category]

    template_text = json.dumps(template, ensure_ascii=False, indent=4)
    return (
        f'{START_SEQ}template\n{template_text}\n{END_SEQ}'
        f'{START_SEQ}user\n{input_text}\n{END_SEQ}'
        f'{START_SEQ}assistant\n'
    )


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.ground_truth and not args.input:
        raise ValueError('--ground-truth requires --input.')

    output = args.output or args.checkpoint.with_name(
        f'{args.checkpoint.stem}-samples.md'
    )
    if output.resolve() == args.checkpoint.resolve():
        raise ValueError('Output cannot overwrite the checkpoint.')

    device = get_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location='cpu',
        weights_only=True,
    )
    model_name = checkpoint['hyper_parameters']['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {
        'cuda': torch.bfloat16,
        'mps': torch.float16,
        'cpu': torch.float32,
    }[device.type]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.load_state_dict(
        {
            key.removeprefix('model.'): value
            for key, value in checkpoint['state_dict'].items()
            if key.startswith('model.')
        }
    )
    model.config.use_cache = True
    model.to(device).eval()
    del checkpoint

    dataset = DATASET[args.dataset_type](
        path=args.data,
        tokenizer=tokenizer,
        template_path=args.template,
        max_length=args.max_length,
    )
    sample_count = min(args.samples, len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), sample_count)

    report = [
        '# Checkpoint sample predictions',
        '',
        f'- Checkpoint: `{args.checkpoint.resolve()}`',
        f'- Dataset: `{args.data.resolve()}`',
        f'- Random seed: `{args.seed}`',
        '',
    ]

    for number, index in enumerate(indices, start=1):
        print(f'Generating sample {number}/{sample_count}', flush=True)
        sample = dataset[index]
        target_start = int(torch.nonzero(sample['labels'] != -100)[0].item())
        prompt_ids = sample['input_ids'][:target_start].to(device)
        attention_mask = sample['attention_mask'][:target_start].to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=prompt_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0]

        prediction_ids = output_ids[prompt_ids.numel() :].cpu()
        target_ids = sample['labels'][sample['labels'] != -100]

        report.extend(
            [
                f'## Sample {number}',
                '',
                f'Dataset index: `{index}`',
                '',
                '### Input',
                '',
                markdown_block(decode(tokenizer, prompt_ids.cpu())),
                '',
                '### Prediction',
                '',
                markdown_block(decode(tokenizer, prediction_ids)),
                '',
                '### Ground truth',
                '',
                markdown_block(decode(tokenizer, target_ids)),
                '',
            ]
        )

    if args.input:
        prompt = custom_prompt(
            args.input,
            args.template,
            args.dataset_type,
            args.category,
        )
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt',
        ).to(device)

        print('Generating custom input', flush=True)
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0]

        prediction_ids = output_ids[encoded['input_ids'].shape[1] :].cpu()
        report.extend(
            [
                '## Custom input',
                '',
                '### Input',
                '',
                markdown_block(prompt),
                '',
                '### Prediction',
                '',
                markdown_block(decode(tokenizer, prediction_ids)),
                '',
                '### Ground truth',
                '',
                markdown_block(args.ground_truth or 'Not provided'),
                '',
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(report), encoding='utf-8')
    print(f'Wrote report: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
