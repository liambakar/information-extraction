import json
from typing import Any, Tuple

from torch.utils.data import Dataset

START_SEQ = '<|im_start|>'
END_SEQ = '<|im_end|>\n'
THINKING_START_SEQ = '<think>'
THINKING_END_SEQ = '</think>'


class FullTemplateInstructionDataset(Dataset):
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

    def _format_message(self, role: str, content: str) -> str:
        return f'{START_SEQ}{role}\n{content}\n{END_SEQ}'

    def _build_instruction(self, row):
        if 'instruction' in row:
            return row['instruction']

        if self.template is None:
            raise ValueError(
                'template_path is required when loading rows with utterance/extraction.'
            )

        template = self._format_json(self.template)

        instruction = (
            self._format_message('template', template)
            + self._format_message('user', row['utterance'])
            + START_SEQ
            + 'assistant'
            + '\n'
        )
        return instruction

    def _build_response(self, row):
        if 'response' in row:
            return self._format_json(row['response']) + '\n' + END_SEQ

        return self._format_json(row['extraction']) + '\n' + END_SEQ

    def __getitem__(self, idx):
        row = self.rows[idx]
        instruction = self._build_instruction(row)
        response = self._build_response(row)
        text = instruction + response

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        encoded_instructions = self.tokenizer(
            instruction,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        encoded_instruction_ids = encoded_instructions['input_ids']

        labels = input_ids.clone()

        # mask the template and user utterance and assistant prefix
        instruction_length = min(len(encoded_instruction_ids), self.max_length)
        labels[:instruction_length] = -100

        # mask the padding
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class PartialTemplateInstructionDataset(Dataset):
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

        template = self._load_template(template_path)

        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                classifications = row['type']
                extractions = row['extraction']
                partial_templates, partial_outputs = self._build_partial_json_template(
                    template, extractions, classifications
                )
                for i, c in enumerate(classifications):
                    self.rows.append(
                        {
                            'index': row['index'],
                            'example_id': row['example_id'],
                            'utterance_id': row['utterance_id'],
                            'classification_id': i,
                            'type': [c],
                            'modality': row['modality'],
                            'utterance': row['utterance'],
                            'template': partial_templates[i],
                            'extraction': partial_outputs[i],
                        }
                    )

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

    def _format_classification_key(self, key):
        mapping = {
            'Symptoms': 'symptom',
            'Treatment': 'treatment',
            'Activities': 'activity',
            'Food': 'food',
            'Mood': 'mood',
        }
        return mapping[key]

    def _build_partial_json_template(
        self, full_template: Any, full_output: Any, classification: list[str]
    ) -> Tuple[list[Any], list[Any]]:
        """Given full template and full outputs, return the ones matching the classification.

        Returns:
            partial_templates, partial_outputs
        """
        partial_templates = []
        partial_outputs = []
        for c in classification:
            partial_templates.append(full_template[self._format_classification_key(c)])
            partial_outputs.append(full_output[self._format_classification_key(c)])
        return partial_templates, partial_outputs

    def _format_message(self, role: str, content: str) -> str:
        return f'{START_SEQ}{role}\n{content}\n{END_SEQ}'

    def _build_instruction(self, row):
        if 'instruction' in row:
            return row['instruction']

        template = self._format_json(row['template'])

        instruction = (
            self._format_message('template', template)
            + self._format_message('user', row['utterance'])
            + START_SEQ
            + 'assistant'
            + '\n'
        )
        return instruction

    def _build_response(self, row):
        if 'response' in row:
            return self._format_json(row['response']) + '\n' + END_SEQ

        return self._format_json(row['extraction']) + '\n' + END_SEQ

    def __getitem__(self, idx):
        row = self.rows[idx]
        instruction = self._build_instruction(row)
        response = self._build_response(row)
        text = instruction + response

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        encoded_instructions = self.tokenizer(
            instruction,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        encoded_instruction_ids = encoded_instructions['input_ids']

        labels = input_ids.clone()

        # mask the template and user utterance and assistant prefix
        instruction_length = min(len(encoded_instruction_ids), self.max_length)
        labels[:instruction_length] = -100

        # mask the padding
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


DATASET = {
    'full': FullTemplateInstructionDataset,
    'partial': PartialTemplateInstructionDataset,
}
