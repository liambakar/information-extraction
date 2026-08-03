from pathlib import Path

from pydantic import ValidationError

from src.training.modules.info_extraction_lightning_module import (
    InfoExtractionModule,
)
from src.training.utils.template_schema import load_template_model


class ConstrainedInfoExtractionModule(InfoExtractionModule):
    def __init__(
        self,
        model_name,
        tokenizer,
        lr,
        weight_decay,
        warmup_steps,
        template_path,
        dataset_type,
        total_steps=None,
    ):
        try:
            import xgrammar as xgr
        except ImportError as error:
            raise ImportError(
                'Constrained decoding requires xgrammar. Install it with '
                '`pip install xgrammar`.'
            ) from error

        super().__init__(
            model_name=model_name,
            tokenizer=tokenizer,
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        self.save_hyperparameters(
            {
                'template_path': str(Path(template_path)),
                'dataset_type': dataset_type,
            }
        )

        self.output_model = load_template_model(template_path, dataset_type)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=self.model.config.vocab_size,
        )
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
        self.compiled_grammar = grammar_compiler.compile_json_schema(self.output_model)

    def _generation_logits_processor(self):
        import xgrammar as xgr

        # XGrammar's HF processor holds matcher state and can only be used once.
        return xgr.contrib.hf.LogitsProcessor(self.compiled_grammar)

    def _validate_generated_ids(self, generated_ids):
        generated_json = self.tokenizer.decode(
            generated_ids.tolist(),
            skip_special_tokens=True,
        ).strip()

        try:
            self.output_model.model_validate_json(generated_json)
        except ValidationError as error:
            raise ValueError(
                'Constrained generation did not produce a complete extraction '
                f'matching the template: {generated_json!r}'
            ) from error
