import torch
import lightning as L
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from lightning.pytorch.utilities.types import OptimizerLRScheduler


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

        return loss

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

        if _batch_idx == 0 and self.trainer.is_global_zero:
            input_ids = batch['input_ids']
            pred_ids = outputs.logits.argmax(dim=-1)

            labels = batch['labels'][i, 1:]
            predictions = pred_ids[i, :-1]

            valid_mask = labels != -100

            target_text = self.tokenizer.decode(
                labels[valid_mask],
                skip_special_tokens=True,
            )
            predicted_text = self.tokenizer.decode(
                predictions[valid_mask],
                skip_special_tokens=True,
            )

            print('\n' + '#' * 40)
            print(f' Validation Samples for Batch {batch_idx} ')
            print('#' * 40)

            for i in range(min(input_ids.size(0), 3)):
                input_text = self.tokenizer.decode(
                    input_ids[i],
                    skip_special_tokens=True,
                )

                predicted_text = self.tokenizer.decode(
                    pred_ids[i, :-1],
                    skip_special_tokens=True,
                )

                target_text = self.tokenizer.decode(
                    input_ids[i, 1:],
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
