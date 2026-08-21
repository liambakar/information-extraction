import json
import math
import unittest

from src.training.modules.rewards import (
    RewardConfig,
    build_reward_functions,
    build_reward_weights,
    evaluate_reward,
    extraction_reward,
    hallucination_reward,
    json_validity_reward,
    reward_function,
    schema_reward,
    total_reward,
)
from src.training.modules.rewards.extraction import score_extraction
from src.training.modules.rewards.structure import score_schema
from src.training.utils.normalization import is_empty, values_match


FULL_EXTRACTION = {
    'activity': {
        'type': 'physical',
        'keywords': ['walking'],
        'duration': '30 minutes',
        'location': 'Park',
        'date': '2026-08-21T09:00:00',
    },
    'food': {
        'consumed_items': ['apple'],
        'missing_items': [],
        'amount': ['one'],
    },
    'mood': {
        'description': 'Good',
        'classification': 'positive',
    },
    'symptom': {
        'keywords': [],
        'description': None,
    },
    'treatment': {
        'name': None,
        'dosage': None,
        'status': None,
    },
}


class RewardTests(unittest.TestCase):
    def test_empty_and_unicode_normalization(self):
        self.assertTrue(is_empty([' ', None]))
        self.assertTrue(values_match('ＣＡＦÉ\t au  lait', 'café au lait'))

    def test_reward_package_uses_working_package_imports(self):
        self.assertTrue(callable(reward_function))

    def test_schema_score_uses_fixed_denominator(self):
        self.assertEqual(score_schema(FULL_EXTRACTION), 1.0)

        missing_activity = dict(FULL_EXTRACTION)
        missing_activity.pop('activity')
        self.assertTrue(math.isclose(score_schema(missing_activity), 18 / 26))

        invalid_activity = dict(FULL_EXTRACTION, activity=None)
        self.assertTrue(math.isclose(score_schema(invalid_activity), 19 / 26))

    def test_partial_schema_score_uses_the_requested_section(self):
        symptom = FULL_EXTRACTION['symptom']

        self.assertEqual(score_schema(symptom, section='symptom'), 1.0)
        self.assertLess(
            score_schema({'keywords': []}, section='symptom'),
            1.0,
        )

    def test_extraction_distinguishes_wrong_values_from_hallucinations(self):
        config = RewardConfig()
        truth = {'section': {'correct': 'yes', 'wrong': 'target', 'empty': None}}
        prediction = {
            'section': {
                'correct': ' YES ',
                'wrong': 'different',
                'empty': 'invented',
                'extra': 'also invented',
            }
        }

        result = score_extraction(prediction, truth, config)

        self.assertEqual(result.correct_values, 1)
        self.assertEqual(result.incorrect_values, 1)
        self.assertEqual(result.hallucinated_values, 2)
        self.assertEqual(result.omissions, 0)
        self.assertTrue(math.isclose(result.extraction, (1.0 - 0.75) / 3))
        self.assertEqual(result.hallucination, -0.5)

    def test_omissions_and_correct_nulls_have_separate_outcomes(self):
        config = RewardConfig()
        truth = {'section': {'present': 'value', 'empty': None}}
        prediction = {'section': {'present': None, 'empty': None}}

        result = score_extraction(prediction, truth, config)

        self.assertEqual(result.omissions, 1)
        self.assertEqual(result.correct_nulls, 1)
        self.assertEqual(result.incorrect_values, 0)
        self.assertEqual(result.hallucinated_values, 0)
        self.assertTrue(math.isclose(result.extraction, (-0.5 + 0.25) / 2))

    def test_list_matching_is_normalized_and_order_independent_by_default(self):
        result = score_extraction(
            {'items': [' Second ', 'FIRST']},
            {'items': ['first', 'second']},
            RewardConfig(),
        )

        self.assertEqual(result.correct_values, 1)
        self.assertEqual(result.incorrect_values, 0)

    def test_invalid_json_returns_only_the_invalid_json_reward(self):
        result = evaluate_reward('not json', FULL_EXTRACTION)

        self.assertEqual(result.total, -1.0)
        self.assertEqual(result.json_validity, -1.0)
        self.assertEqual(result.schema, 0.0)
        self.assertEqual(result.extraction, 0.0)
        self.assertEqual(result.hallucination, 0.0)

    def test_invalid_ground_truth_fails_fast(self):
        with self.assertRaisesRegex(ValueError, 'ground_truth is not valid JSON'):
            evaluate_reward('{}', 'not json')

    def test_component_rewards_sum_to_total(self):
        prediction = json.dumps(FULL_EXTRACTION)
        truth = json.dumps(FULL_EXTRACTION)

        components = [
            json_validity_reward([prediction])[0],
            schema_reward([prediction])[0],
            extraction_reward([prediction], [truth])[0],
            hallucination_reward([prediction], [truth])[0],
        ]

        weighted_total = sum(
            component * weight
            for component, weight in zip(
                components,
                build_reward_weights(RewardConfig()),
            )
        )
        self.assertTrue(
            math.isclose(weighted_total, total_reward([prediction], [truth])[0])
        )

    def test_conversational_completions_are_supported(self):
        prediction = json.dumps(FULL_EXTRACTION)
        completions = [[{'role': 'assistant', 'content': prediction}]]

        self.assertEqual(json_validity_reward(completions), [0.1])
        self.assertEqual(
            total_reward(completions, [FULL_EXTRACTION]),
            [reward_function(prediction, FULL_EXTRACTION)],
        )

    def test_schema_reward_supports_partial_dataset_metadata(self):
        symptom = json.dumps(FULL_EXTRACTION['symptom'])

        self.assertEqual(
            schema_reward([symptom], schema_section=['symptom']),
            [1.0],
        )
        components = [
            json_validity_reward([symptom])[0],
            schema_reward([symptom], schema_section=['symptom'])[0],
            extraction_reward([symptom], [symptom])[0],
            hallucination_reward([symptom], [symptom])[0],
        ]
        weighted_total = sum(
            component * weight
            for component, weight in zip(
                components,
                build_reward_weights(RewardConfig()),
            )
        )
        self.assertTrue(
            math.isclose(
                weighted_total,
                total_reward(
                    [symptom],
                    [symptom],
                    schema_section=['symptom'],
                )[0],
            )
        )

    def test_schema_reward_logs_ground_truth_with_completion_rows(self):
        logged_columns = {}
        prediction = json.dumps(FULL_EXTRACTION)
        ground_truth = json.dumps(FULL_EXTRACTION)

        schema_reward(
            [prediction],
            ground_truth=[ground_truth],
            schema_section=[''],
            log_extra=lambda name, values: logged_columns.update({name: values}),
        )

        self.assertEqual(logged_columns['ground_truth'], [ground_truth])

    def test_bound_reward_functions_preserve_names_and_config(self):
        config = RewardConfig(valid_json=0.25)
        reward_funcs = build_reward_functions(config)

        self.assertEqual(
            [reward_func.__name__ for reward_func in reward_funcs],
            [
                'json_validity_reward',
                'schema_reward',
                'extraction_reward',
                'hallucination_reward',
            ],
        )
        self.assertEqual(reward_funcs[0](['{}']), [0.25])


if __name__ == '__main__':
    unittest.main()
