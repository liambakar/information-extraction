import json

from torch.utils.data import Dataset


class InstructionDataset(Dataset):
    def __init__(
        self,
        path,
        tokenizer,
        template_path,
        max_length=2048,
    ):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = self._load_template(template_path)

        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def _load_template(self, template_path):
        if template_path is None:
            return None

        with open(template_path, 'r') as f:
            return json.load(f)

    def _format_json(self, value):
        if isinstance(value, str):
            return value

        return json.dumps(value, indent=4, ensure_ascii=False)

    def _build_instruction(self, row):
        if 'instruction' in row:
            return row['instruction']

        if self.template is None:
            raise ValueError(
                'template_path is required when loading rows with utterance/extraction.'
            )

        template = self._format_json(self.template)

        return (
            f'<|input|>\n'
            f'### Template:\n{template}\n'
            f'### Text:\n{row["utterance"]}\n\n'
            f'<|output|>'
        )

    def _build_response(self, row):
        if 'response' in row:
            return self._format_json(row['response'])

        return self._format_json(row['extraction'])

    def __getitem__(self, idx):
        row = self.rows[idx]

        messages = [
            {'role': 'user', 'content': self._build_instruction(row)},
            {'role': 'assistant', 'content': self._build_response(row)},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
