import lightning as L
from pathlib import Path
from typing import Any, Protocol, cast


class SavePretrained(Protocol):
    def save_pretrained(
        self,
        save_directory: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...


class SaveHFModelCallback(L.Callback):
    """
    Saves Hugging Face-format checkpoints for inference.

    This is not what you resume from.
    Resume from Lightning .ckpt files instead.
    """

    def __init__(self, output_dir: str | Path):
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_train_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if not trainer.is_global_zero:
            return

        save_dir = self.output_dir / 'final'
        save_dir.mkdir(parents=True, exist_ok=True)

        model = getattr(pl_module, 'model')
        model = getattr(model, 'module', model)

        cast(SavePretrained, model).save_pretrained(save_dir)

        tokenizer = getattr(pl_module, 'tokenizer', None)
        if tokenizer is not None:
            cast(SavePretrained, tokenizer).save_pretrained(save_dir)
