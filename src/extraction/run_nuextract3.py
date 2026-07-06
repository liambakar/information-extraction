import argparse
import json

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL_ID = 'numind/NuExtract3'
DEFAULT_TEMPLATE_PATH = 'extraction_templates/template.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run NuExtract3 structured extraction on text using CUDA.'
    )
    parser.add_argument('--template_path', default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument('--text', required=True)
    parser.add_argument('--model_id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--enable_thinking', action='store_true')

    return parser.parse_args()


def load_model(model_id=DEFAULT_MODEL_ID):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to run NuExtract3 with this script.')

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={'': 'cuda:0'},
        trust_remote_code=True,
    ).eval()
    return processor, model


def load_template(template_path=DEFAULT_TEMPLATE_PATH):
    with open(template_path) as template_file:
        return json.load(template_file)


def run_extraction(
    text,
    model,
    processor,
    template,
    temperature=0.2,
    max_new_tokens=4096,
    enable_thinking=False,
):
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': text,
                }
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
        template=json.dumps(template, indent=4),
        enable_thinking=enable_thinking,
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(  # type: ignore
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )

    generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return output


def main():
    args = parse_args()
    processor, model = load_model(args.model_id)
    template = load_template(args.template_path)
    output = run_extraction(
        text=args.text,
        model=model,
        processor=processor,
        template=template,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
    )
    print(output)
    return output


if __name__ == '__main__':
    main()
