import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.training.modules.rewards import DEFAULT_REWARD_CONFIG, evaluate_reward
from src.training.modules.rewards.evaluator import parse_json_object
from src.training.modules.rewards.models import RewardConfig
from src.training.train_rl import build_prompt, load_jsonl, load_reward_config
from src.training.utils.normalization import flatten_leaves, is_empty, values_match


DEFAULT_DATASET = 'datasets/nuextract3_validation_utterance_outputs.jsonl'
DEFAULT_TEMPLATE = 'extraction_templates/template.json'
DEFAULT_MODELS = {
    'finetuned': 'lbakar/health-log-extraction',
    'qwen': 'Qwen/Qwen3-0.6B',
    'qwen-vl': 'Qwen/Qwen3.5-4B',
    'nuextract': 'numind/NuExtract3',
}
QWEN_SYSTEM_PROMPT = (
    'Extract structured health information from the user text. Return only one '
    'valid JSON object with exactly the requested keys. Use null for a missing '
    'scalar and [] for a missing list. Do not invent unsupported information.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Generate health-log extractions and score them against JSONL labels.'
        )
    )
    parser.add_argument(
        '--mode',
        choices=DEFAULT_MODELS,
        default='finetuned',
        help='Select the model-specific prompt and loading protocol.',
    )
    parser.add_argument(
        '--model',
        help='Override the selected mode\'s default Hugging Face model.',
    )
    parser.add_argument('--dataset', default=DEFAULT_DATASET)
    parser.add_argument('--template', default=DEFAULT_TEMPLATE)
    parser.add_argument('--reward-config')
    parser.add_argument('--output')
    parser.add_argument('--max-rows', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
    )
    return parser.parse_args()


def qwen_messages(
    template: dict[str, Any], utterance: str
) -> list[dict[str, str]]:
    return [
        {'role': 'system', 'content': QWEN_SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': (
                f'Template:\n{json.dumps(template, indent=4, ensure_ascii=False)}'
                f'\n\nHealth log:\n{utterance}'
            ),
        },
    ]


def generate_causal_predictions(
    model_name: str,
    rows: list[dict[str, Any]],
    template: dict[str, Any],
    *,
    mode: str,
    batch_size: int,
    max_new_tokens: int,
    device_name: str,
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device_name == 'auto':
        device_name = (
            'cuda'
            if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available()
            else 'cpu'
        )

    print(f'Loading {model_name} on {device_name}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype='auto').to(
        device_name
    )
    model.eval()

    prompts = (
        [build_prompt(template, row['utterance']) for row in rows]
        if mode == 'finetuned'
        else [
            tokenizer.apply_chat_template(
                qwen_messages(template, row['utterance']),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in rows
        ]
    )
    predictions = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            add_special_tokens=False,
        ).to(device_name)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[:, inputs['input_ids'].shape[1] :]
        predictions.extend(
            tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        )
        print(
            f'Generated {min(start + batch_size, len(prompts))}/{len(prompts)}',
            flush=True,
        )

    return [prediction.strip() for prediction in predictions]


def generate_nuextract_predictions(
    model_name: str,
    rows: list[dict[str, Any]],
    template: dict[str, Any],
    *,
    mode: str,
    max_new_tokens: int,
    device_name: str,
) -> list[str]:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if device_name == 'auto':
        device_name = (
            'cuda'
            if torch.cuda.is_available()
            else 'mps'
            if torch.backends.mps.is_available()
            else 'cpu'
        )

    print(f'Loading {model_name} on {device_name}...', flush=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        dtype='auto',
        trust_remote_code=True,
    ).to(device_name)
    model.eval()

    template_text = json.dumps(template, indent=4, ensure_ascii=False)
    predictions = []
    for index, row in enumerate(rows, start=1):
        if mode == 'nuextract':
            messages = [
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': row['utterance']}],
                }
            ]
            template_args = {'template': template_text}
        else:
            messages = [
                {
                    'role': message['role'],
                    'content': [{'type': 'text', 'text': message['content']}],
                }
                for message in qwen_messages(template, row['utterance'])
            ]
            template_args = {}
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
            enable_thinking=False,
            **template_args,
        ).to(device_name)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[:, inputs['input_ids'].shape[1] :]
        predictions.append(
            processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        )
        print(f'Generated {index}/{len(rows)}', flush=True)

    return predictions


def summarize(
    rows: list[dict[str, Any]],
    predictions: list[str],
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(rows) != len(predictions):
        raise ValueError('Each dataset row must have exactly one prediction.')

    totals = {
        'total': 0.0,
        'json_validity': 0.0,
        'schema': 0.0,
        'extraction': 0.0,
        'hallucination': 0.0,
    }
    counts = {
        'correct_values': 0,
        'correct_nulls': 0,
        'omissions': 0,
        'incorrect_values': 0,
        'hallucinated_values': 0,
        'predicted_fields': 0,
    }
    valid_json = exact_matches = schema_matches = 0
    truth_fields = non_empty_truth_fields = 0
    details = []

    for row, prediction in zip(rows, predictions):
        truth = row['extraction']
        truth_leaves = flatten_leaves(truth)
        truth_fields += len(truth_leaves)
        non_empty_truth_fields += sum(
            not is_empty(value) for value in truth_leaves.values()
        )
        reward = evaluate_reward(
            prediction, truth, config, utterance=row['utterance'],
        )

        for name in totals:
            totals[name] += getattr(reward, name)
        for name in counts:
            counts[name] += getattr(reward, name)

        try:
            parsed = parse_json_object(prediction, name='prediction')
        except ValueError:
            parsed = None
        else:
            valid_json += 1
            schema_matches += reward.schema == 1.0
            exact_matches += values_match(
                parsed,
                truth,
                case_sensitive=config.case_sensitive,
                list_order_sensitive=config.list_order_sensitive,
            )

        details.append(
            {
                **{
                    key: row.get(key)
                    for key in ('index', 'example_id', 'utterance_id')
                },
                'utterance': row['utterance'],
                'ground_truth': truth,
                'prediction': parsed if parsed is not None else prediction,
                'rewards': asdict(reward),
            }
        )

    examples = len(rows)
    correct_leaves = counts['correct_values'] + counts['correct_nulls']
    precision = (
        counts['correct_values'] / counts['predicted_fields']
        if counts['predicted_fields']
        else 0.0
    )
    recall = (
        counts['correct_values'] / non_empty_truth_fields
        if non_empty_truth_fields
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    weights = {
        'json_validity': config.json_validity_weight,
        'schema': config.schema_weight,
        'extraction': config.extraction_weight,
        'hallucination': config.hallucination_weight,
    }
    reward_means = {name: value / examples for name, value in totals.items()}

    summary = {
        'examples': examples,
        'reward_config': asdict(config),
        'rewards': {
            name: {
                'mean': reward_means[name],
                'weight': weight,
                'weighted_mean': reward_means[name] * weight,
            }
            for name, weight in weights.items()
        },
        'mean_total_reward': reward_means['total'],
        'accuracy': {
            'exact_match': exact_matches / examples,
            'leaf_accuracy': correct_leaves / truth_fields,
            'non_empty_precision': precision,
            'non_empty_recall': recall,
            'non_empty_f1': f1,
            'valid_json_rate': valid_json / examples,
            'exact_schema_rate': schema_matches / examples,
        },
        'counts': {
            **counts,
            'ground_truth_fields': truth_fields,
            'non_empty_ground_truth_fields': non_empty_truth_fields,
            'exact_matches': exact_matches,
            'valid_json': valid_json,
            'exact_schema': schema_matches,
        },
    }
    return summary, details


def print_summary(summary: dict[str, Any]) -> None:
    print('\nValidation summary')
    print(f"Examples: {summary['examples']}")
    print('\nRewards (mean, weight, weighted mean)')
    for name, values in summary['rewards'].items():
        print(
            f"  {name:16} {values['mean']:8.4f}  "
            f"x {values['weight']:.2f} = {values['weighted_mean']:8.4f}"
        )
    print(f"  {'total':16} {summary['mean_total_reward']:8.4f}")
    print('\nAccuracy')
    for name, value in summary['accuracy'].items():
        print(f'  {name:24} {value:8.2%}')
    print('\nExtraction outcomes')
    for name, value in summary['counts'].items():
        print(f'  {name:32} {value}')


def main() -> None:
    args = parse_args()
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else 1
        if args.mode in ('nuextract', 'qwen-vl')
        else 4
    )
    if batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError('--batch-size and --max-new-tokens must be positive.')
    if args.mode in ('nuextract', 'qwen-vl') and batch_size != 1:
        raise ValueError('Multimodal validation currently requires --batch-size 1.')

    rows = load_jsonl(args.dataset)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError('--max-rows must be positive.')
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError('The validation dataset is empty.')
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row.get('utterance'), str) or not isinstance(
            row.get('extraction'), dict
        ):
            raise ValueError(
                f'Dataset row {line_number} needs an utterance string and '
                'an extraction object.'
            )

    with Path(args.template).open(encoding='utf-8') as template_file:
        template = json.load(template_file)
    config = (
        load_reward_config(args.reward_config)
        if args.reward_config
        else DEFAULT_REWARD_CONFIG
    )
    model_name = args.model or DEFAULT_MODELS[args.mode]
    if args.mode in ('nuextract', 'qwen-vl'):
        predictions = generate_nuextract_predictions(
            model_name,
            rows,
            template,
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            device_name=args.device,
        )
    else:
        predictions = generate_causal_predictions(
            model_name,
            rows,
            template,
            mode=args.mode,
            batch_size=batch_size,
            max_new_tokens=args.max_new_tokens,
            device_name=args.device,
        )
    summary, details = summarize(rows, predictions, config)
    print_summary(summary)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as output_file:
            json.dump(
                {
                    'mode': args.mode,
                    'model': model_name,
                    'summary': summary,
                    'examples': details,
                },
                output_file,
                indent=2,
                ensure_ascii=False,
            )
        print(f'\nSaved report to {output_path}')


if __name__ == '__main__':
    main()
