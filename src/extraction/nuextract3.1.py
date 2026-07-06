import json

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = 'numind/NuExtract3'

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
).eval()


def run_nuextract(messages, **chat_template_kwargs):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
        **chat_template_kwargs,
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

    generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


# Single image structured extraction
receipt_image = Image.open('receipt.png').convert('RGB')
receipt_messages = [
    {
        'role': 'user',
        'content': [
            {
                'type': 'image',
                'image': receipt_image,
            }
        ],
    }
]

template = {
    'store': 'verbatim-string',
    'date': 'date-time',
    'total': 'number',
    'payment_method': 'verbatim-string',
}

structured_output = run_nuextract(
    receipt_messages,
    template=json.dumps(template, indent=4),
    enable_thinking=False,
)
print(structured_output)

# Single image content extraction
document_image = Image.open('document.png').convert('RGB')
document_messages = [
    {
        'role': 'user',
        'content': [
            {
                'type': 'image',
                'image': document_image,
            }
        ],
    }
]

content_output = run_nuextract(
    document_messages,
    mode='content',
    enable_thinking=False,
)
print(content_output)
