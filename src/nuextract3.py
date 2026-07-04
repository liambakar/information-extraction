from mlx_vlm import load, generate
from mlx_vlm.utils import load_image

# Load the 8-bit quantized MLX model
model_id = 'numind/NuExtract3-mlx-4bits'
model, processor = load(model_id)

# Define your JSON structure template
json_template = """
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

# Format prompt according to NuExtract3's extraction structure
prompt = (
    f'Extract information from the document matching this JSON schema:\n{json_template}'
)

# Load your document image (e.g., receipt, scan, invoice)
# image = load_image('path/to/invoice.jpg')

# Generate structured data
output = generate(model, processor, prompt, verbose=True)
print(output)
