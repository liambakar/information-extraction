# information extraction inference

`./scripts/submit_extraction_job.sh nuextract3_run out/slurm_logs 8 datasets/preprocessed_dataset_claude_5_tones.jsonl out/nuextract3_outputs.jsonl`

`./scripts/submit_extraction_job.sh nuextract3_run out/slurm_logs 8 datasets/preprocessed_dataset_claude_5_tones.jsonl out/nuextract3_outputs.jsonl --reset-scratch`

# information extraction training

`./scripts/submit_train_job.sh configs/qwen_training_config.json`

`./scripts/submit_rl_train_job.sh configs/qwen_rl_training_config.json`

The RL hallucination reward compares each populated extraction value with the
original utterance using the three-way NLI model
[`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).
The model downloads on first use and runs on the current training GPU. Confident
contradictions receive the full configured hallucination penalty; confident
neutral judgments (unsupported claims) receive half. Entailed claims and
low-confidence judgments receive no hallucination penalty. Adjust
`rewards.nli_confidence_threshold` in the RL config after checking judgments on
labeled examples from this dataset.
