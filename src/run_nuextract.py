import json
import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Arguments for running NuExtract Model.'
    )

    parser.add_argument('--checkpoint_path', default='numind/NuExtract-tiny-v1.5')
    parser.add_argument('--template_path', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--temperature', default=1.0)

    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    args = parse_args()

    CHECKPOINT_PATH = args.checkpoint_path
    DEVICE = args.device if args.device else get_device()
    TEMPLATE_PATH = args.template_path
    TEMPERATURE = args.temperature

    model = (
        AutoModelForCausalLM.from_pretrained(
            CHECKPOINT_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to(DEVICE)
        .eval()
    )

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)

    with open(TEMPLATE_PATH) as f:
        template = json.load(f)

    text = input(
        'What would you like to extract information from? \nWrite it on this line: '
    )

    prompt = (
        f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>"""
    )

    with torch.no_grad():
        encoding = tokenizer(
            [prompt],
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=10_000,
        ).to(DEVICE)

        prediction = model.generate(
            **encoding, temperature=TEMPERATURE, max_new_tokens=4_000
        )

        output = tokenizer.batch_decode(prediction, skip_special_tokens=True)

        print(output[0].split('<|output|>')[1])


if __name__ == '__main__':
    main()
