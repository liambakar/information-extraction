import torch
import lightning as L
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from lightning.pytorch.utilities.types import OptimizerLRScheduler


class InfoExtractionModule(L.LightningModule):
    def __init__(
        self,
        model_name,
        lr=1e-5,
        weight_decay=0.01,
        warmup_steps=100,
        total_steps=None,
    ):
        super().__init__()
        self.save_hyperparameters()

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
