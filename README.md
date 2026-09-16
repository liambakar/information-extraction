# information extraction inference

`./scripts/submit_extraction_job.sh nuextract3_run out/slurm_logs 8 datasets/preprocessed_dataset_claude_5_tones.jsonl out/nuextract3_outputs.jsonl`

`./scripts/submit_extraction_job.sh nuextract3_run out/slurm_logs 8 datasets/preprocessed_dataset_claude_5_tones.jsonl out/nuextract3_outputs.jsonl --reset-scratch`

# information extraction training

`./scripts/submit_train_job.sh configs/qwen_training_config.json`

`./scripts/submit_rl_train_job.sh configs/qwen_rl_training_config.json`
