import torch
import lightning as L
from transformers import (
    AutoModelForCausalLM,
    LogitsProcessor,
    get_cosine_schedule_with_warmup,
)
from lightning.pytorch.utilities.types import OptimizerLRScheduler


MAX_LOGGED_EXAMPLES = 4
MAX_LOGGED_NEW_TOKENS = 128


class InfoExtractionModule(L.LightningModule):
    def __init__(
        self,
        model_name,
        tokenizer,
        lr,
        weight_decay,
        warmup_steps,
        total_steps=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.tokenizer = tokenizer

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        self.example_batch = None

    def training_step(self, batch, _batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss

        self.log(
            'train_loss',
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        if _batch_idx == 0 and self.trainer.is_global_zero:
            input_ids = batch['input_ids']
            pred_ids = outputs.logits.argmax(dim=-1)

            print('\n' + '#' * 40)
            print(f' Training Samples for Batch {_batch_idx} ')
            print('#' * 40)

            for i in range(min(input_ids.size(0), 3)):
                labels = batch['labels'][i, 1:]
                predictions = pred_ids[i, :-1]
                valid_mask = labels != -100

                input_text = self.tokenizer.decode(
                    input_ids[i],
                    skip_special_tokens=True,
                )

                predicted_text = self.tokenizer.decode(
                    predictions[valid_mask],
                    skip_special_tokens=True,
                )

                target_text = self.tokenizer.decode(
                    labels[valid_mask],
                    skip_special_tokens=True,
                )

                print(f'--- SAMPLE {i + 1} ---')
                print(f'INPUT TEXT:\n{input_text.strip()}')
                print(f'\nTARGET NEXT TOKENS:\n{target_text.strip()}')
                print(f'\nPREDICTED NEXT TOKENS:\n{predicted_text.strip()}')
                print('-' * 20)
                print('\n')

            print('#' * 40 + '\n')

        return loss

    def on_validation_start(self):
        if self.trainer.is_global_zero:
            self.example_batch = None

    def validation_step(self, batch, _batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss

        self.log(
            'val_loss',
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if (
            _batch_idx == 0
            and self.trainer.is_global_zero
            and self.example_batch is None
        ):
            self.example_batch = self._capture_example_batch(batch)

        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return

        batch = self.example_batch
        self.example_batch = None

        table_logger = self._get_table_logger()
        if batch is None or table_logger is None:
            return

        if rows := self._build_example_rows(batch):
            table_logger.log_table(
                key='examples',
                columns=['input', 'prediction', 'ground_truth'],
                data=rows,
            )

    def _capture_example_batch(self, batch):
        return {key: self._capture_value(key, value) for key, value in batch.items()}

    def _get_table_logger(self):
        for logger in [self.logger, *getattr(self, 'loggers', [])]:
            if hasattr(logger, 'log_table'):
                return logger

        return None

    def _capture_value(self, key, value):
        if isinstance(value, torch.Tensor):
            return value[:MAX_LOGGED_EXAMPLES].detach().cpu().clone()

        if isinstance(value, list):
            return value[:MAX_LOGGED_EXAMPLES]

        raise ValueError(f'Unsupported type in batch for key {key}: {type(value)}')

    def _build_example_rows(self, batch):
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch.get('attention_mask')
        rows = []

        for i in range(input_ids.size(0)):
            label_positions = torch.nonzero(labels[i] != -100, as_tuple=False)
            if label_positions.numel() == 0:
                continue

            target_start = int(label_positions[0].item())
            if target_start == 0:
                continue

            prompt_ids = input_ids[i, :target_start].to(self.device)
            generation_inputs = {
                'input_ids': prompt_ids.unsqueeze(0),
            }

            if attention_mask is not None:
                generation_inputs['attention_mask'] = (
                    attention_mask[i, :target_start].to(self.device).unsqueeze(0)
                )

            with torch.no_grad():
                generation_kwargs = {
                    **generation_inputs,
                    'max_new_tokens': MAX_LOGGED_NEW_TOKENS,
                    'do_sample': False,
                    'pad_token_id': self._generation_pad_token_id(),
                }
                if logits_processor := self._generation_logits_processor():
                    generation_kwargs['logits_processor'] = [logits_processor]

                output = self.model.generate(**generation_kwargs)  # type: ignore

            generated_ids = output[0, prompt_ids.size(0) :].detach().cpu()
            self._validate_generated_ids(generated_ids)
            target_ids = labels[i][labels[i] != -100]

            rows.append(
                [
                    self._decode_log_tokens(prompt_ids.detach().cpu()),
                    self._decode_log_tokens(generated_ids),
                    self._decode_log_tokens(target_ids),
                ]
            )

        return rows

    def _generation_logits_processor(self) -> None | LogitsProcessor:
        return None

    def _validate_generated_ids(self, _generated_ids):
        return None

    def _decode_log_tokens(self, token_ids):
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            token_ids = token_ids[token_ids != pad_token_id]

        return self.tokenizer.decode(
            token_ids.tolist(),
            skip_special_tokens=False,
        ).strip()

    def _generation_pad_token_id(self):
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id

        return self.tokenizer.eos_token_id

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        total_steps = self.total_steps

        if total_steps is None:
            total_steps = self.trainer.estimated_stepping_batches

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=int(total_steps),
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }
