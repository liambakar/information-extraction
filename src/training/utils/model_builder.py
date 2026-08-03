from typing import Literal

from src.training.modules.info_extraction_lightning_module import InfoExtractionModule
from src.training.modules.constrained_info_extraction_lightning_module import (
    ConstrainedInfoExtractionModule,
)

MODULE_TYPES = {
    'standard': InfoExtractionModule,
    'constrained': ConstrainedInfoExtractionModule,
}


def build_model(
    model_name: str,
    module_type: Literal['standard', 'constrained'],
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    tokenizer,
    template_path,
    dataset_type,
):
    module_class = MODULE_TYPES[module_type]
    module_kwargs = {
        'model_name': model_name,
        'lr': lr,
        'weight_decay': weight_decay,
        'warmup_steps': warmup_steps,
        'tokenizer': tokenizer,
    }
    if module_type == 'constrained':
        module_kwargs.update(
            template_path=template_path,
            dataset_type=dataset_type,
        )

    model = module_class(**module_kwargs)
    return model
