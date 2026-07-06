from mlx_vlm import load, generate
import json
# from mlx_vlm.utils import load_image

# Load the 8-bit quantized MLX model
model_id = 'numind/NuExtract3-mlx-4bits'
model, processor = load(model_id)

# Define your JSON structure template
template = """
{
  "invoice_number": "verbatim-string",
  "invoice_date": "date",
  "total_amount": "number",
  "currency": "currency",
  "line_items": [
    {
      "description": "verbatim-string",
      "item_type": ["electronics", "clothing", "vehicle", "furniture", "other"],
      "quantity": "integer",
      "unit_price": "number",
      "total": "number"
    }
  ]
}
"""


text = (
    'I bought 3 apples at $10 each. This happened yesterday. Today is august 31st 2021.'
)
prompt = f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>"""


# Load your document image (e.g., receipt, scan, invoice)
# image = load_image('path/to/invoice.jpg')

# Generate structured data
output = generate(model, processor, prompt, verbose=False)
output_token = '<|output|>'

outputs = output.text.split(output_token)

for o in outputs:
    if output_token not in o:
        print(o)
