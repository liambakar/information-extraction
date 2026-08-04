import json
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    training_dataset_path: str
    dataset_type: Literal['full', 'partial']
    output_directory: str
    extraction_template_path: str
    num_workers: int
    max_rows: int
    validation_split: float = Field(ge=0.0, lt=1.0)


class ModelConfig(BaseModel):
    model_name: str
    max_length: int
    lightning_module_type: Literal['standard', 'constrained'] = 'standard'


class OptimizationConfig(BaseModel):
    lr: float
    weight_decay: float
    warmup_steps: int
    batch_size: int
    accumulate_grad_batches: int
    max_epochs: int
    patience: int = Field(ge=0)


class HardwareConfig(BaseModel):
    devices: str | int  # Handles "auto" or specific GPU counts like 1
    num_nodes: int


class WandbConfig(BaseModel):
    use_wandb: bool
    project: str
    run_name: Optional[str] = None  # Allows null/None
    entity: Optional[str] = None  # Allows null/None
    log_model: bool


class CheckpointingConfig(BaseModel):
    resume_from: Optional[str] = None


class TrainConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    optimization: OptimizationConfig
    hardware: HardwareConfig
    wandb: WandbConfig
    checkpointing: CheckpointingConfig

    def __init__(self, filepath_or_data: str | dict[str, Any], **kwargs):
        if isinstance(filepath_or_data, str):
            with open(filepath_or_data, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            super().__init__(**file_data)
        elif isinstance(filepath_or_data, dict):
            super().__init__(**filepath_or_data)
        else:
            super().__init__(**kwargs)
