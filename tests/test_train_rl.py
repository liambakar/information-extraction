import json
import tempfile
import unittest
from pathlib import Path

from src.training.utils.config_parser import TrainConfig
from src.training.train_rl import (
    build_prompt,
    build_rl_records,
    find_last_checkpoint,
    load_reward_config,
    load_rl_settings,
    select_rows,
    split_records,
    validate_completion_logging_support,
    validate_training_dependency_versions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


TEMPLATE = {
    'activity': {'type': ['physical'], 'keywords': ['string']},
    'symptom': {'keywords': ['string'], 'description': 'string'},
}

ROW = {
    'utterance': 'My knee hurts after a walk.',
    'type': ['Symptoms', 'Activities'],
    'extraction': {
        'activity': {'type': 'physical', 'keywords': ['walk']},
        'symptom': {'keywords': ['knee'], 'description': 'Knee hurts.'},
    },
}


class RLTrainingHelpersTest(unittest.TestCase):
    def test_tested_rl_dependency_versions_are_accepted(self):
        validate_training_dependency_versions(
            {
                'torch': '2.11.0+cu128',
                'transformers': '5.13.1',
                'trl': '1.10.0',
                'datasets': '4.7.0',
                'accelerate': '1.12.0',
            }
        )

    def test_incompatible_rl_dependencies_have_repair_command(self):
        versions = {
            'torch': '2.11.0+cu128',
            'transformers': '5.16.1',
            'trl': '0.12.1',
            'datasets': '2.19.1',
            'accelerate': 'not installed',
        }

        with self.assertRaisesRegex(
            RuntimeError,
            r'(?s)trl.*0\.12\.1.*setup_rl_environment\.sh',
        ):
            validate_training_dependency_versions(versions)

    def test_dedicated_rl_config_loads_all_training_sections(self):
        config_path = REPO_ROOT / 'configs' / 'qwen_rl_training_config.json'

        train_config = TrainConfig(str(config_path))
        rl_settings = load_rl_settings(config_path)
        reward_config = load_reward_config(str(config_path))

        self.assertEqual(train_config.data.dataset_type, 'full')
        self.assertEqual(rl_settings.num_generations, 4)
        self.assertEqual(rl_settings.beta, 0.001)
        self.assertEqual(train_config.checkpointing.save_steps, 10)
        self.assertEqual(reward_config.extraction_weight, 0.7)

    def test_prompt_matches_existing_qwen_message_format(self):
        prompt = build_prompt(TEMPLATE, ROW['utterance'])

        self.assertIn('<|im_start|>template\n', prompt)
        self.assertIn('<|im_start|>user\nMy knee hurts after a walk.', prompt)
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n'))

    def test_full_records_keep_the_complete_template_and_target(self):
        records = build_rl_records([ROW], TEMPLATE, 'full')

        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0]['ground_truth']), ROW['extraction'])
        self.assertEqual(records[0]['schema_section'], '')

    def test_partial_records_expand_each_classification(self):
        records = build_rl_records([ROW], TEMPLATE, 'partial')

        self.assertEqual(len(records), 2)
        self.assertEqual(
            [record['schema_section'] for record in records],
            ['symptom', 'activity'],
        )
        self.assertEqual(
            json.loads(records[0]['ground_truth']),
            ROW['extraction']['symptom'],
        )
        self.assertNotIn('"activity"', records[0]['prompt'])

    def test_row_selection_and_splitting_are_reproducible(self):
        rows = [{'row': index} for index in range(10)]
        selected_once = select_rows(rows, max_rows=5, seed=42)
        selected_twice = select_rows(rows, max_rows=5, seed=42)
        self.assertEqual(selected_once, selected_twice)

        records = [
            {'prompt': str(index), 'ground_truth': '{}', 'schema_section': ''}
            for index in range(10)
        ]
        first_split = split_records(records, validation_split=0.2, seed=42)
        second_split = split_records(records, validation_split=0.2, seed=42)
        self.assertEqual(first_split, second_split)
        self.assertEqual(len(first_split[0]), 8)
        self.assertEqual(len(first_split[1]), 2)

    def test_reward_config_requires_all_nested_reward_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'rewards.json'
            values = json.loads(
                (REPO_ROOT / 'configs' / 'qwen_rl_training_config.json').read_text()
            )['rewards']
            values['correct_value'] = 2.0
            path.write_text(json.dumps({'rewards': values}), encoding='utf-8')

            config = load_reward_config(str(path))

        self.assertEqual(config.correct_value, 2.0)

    def test_missing_reward_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'rewards.json'
            path.write_text(json.dumps({'rewards': {'correct_value': 2.0}}))

            with self.assertRaisesRegex(ValueError, 'Invalid reward config'):
                load_reward_config(path)

    def test_missing_rl_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'training.json'
            path.write_text(json.dumps({'rl': {'num_generations': 4}}))

            with self.assertRaisesRegex(ValueError, 'Invalid rl config'):
                load_rl_settings(path)

    def test_last_checkpoint_uses_the_largest_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / 'checkpoint-20').mkdir()
            (output_dir / 'checkpoint-100').mkdir()
            (output_dir / 'checkpoints').mkdir()

            checkpoint = find_last_checkpoint(output_dir)

        self.assertTrue(checkpoint.endswith('checkpoint-100'))

    def test_completion_logging_support_fails_fast_for_old_trl(self):
        class OldConfig:
            __dataclass_fields__ = {}

        class OldTrainer:
            pass

        with self.assertRaisesRegex(RuntimeError, 'Upgrade TRL'):
            validate_completion_logging_support(OldConfig, OldTrainer)

    def test_completion_logging_support_accepts_expected_trl_api(self):
        class CurrentConfig:
            __dataclass_fields__ = {
                'log_completions': object(),
                'num_completions_to_print': object(),
                'log_unique_prompts': object(),
            }

        class CurrentTrainer:
            def _log_completion_extra(self):
                pass

        validate_completion_logging_support(CurrentConfig, CurrentTrainer)


if __name__ == '__main__':
    unittest.main()
