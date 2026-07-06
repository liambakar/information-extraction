#!/bin/bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-datasets/preprocessed_dataset_claude_5_tones.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-out/nuextract3_outputs.jsonl}"
EXTRACT_SCRIPT="${EXTRACT_SCRIPT:-src/extraction/run_nuextract3.py}"
STATE_SCRIPT="${STATE_SCRIPT:-src/worker/resume.py}"
CLAIM_TIMEOUT_SECONDS="${CLAIM_TIMEOUT_SECONDS:-3600}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

EXTRACTION_PATH=""
cleanup() {
    if [[ -n "$EXTRACTION_PATH" && -f "$EXTRACTION_PATH" ]]; then
        rm -f "$EXTRACTION_PATH"
    fi
}
trap cleanup EXIT

while true; do
    set +e
    CLAIMED_JSON="$(
        python "$STATE_SCRIPT" claim-next \
            --dataset_path "$DATASET_PATH" \
            --claim_timeout_seconds "$CLAIM_TIMEOUT_SECONDS"
    )"
    CLAIM_STATUS=$?
    set -e

    if [[ "$CLAIM_STATUS" -eq 2 ]]; then
        echo "No unprocessed rows remain."
        exit 0
    fi

    if [[ "$CLAIM_STATUS" -ne 0 ]]; then
        echo "Failed to claim next row." >&2
        exit "$CLAIM_STATUS"
    fi

    INDEX="$(
        python -c 'import json, sys; print(json.loads(sys.argv[1])["index"])' \
            "$CLAIMED_JSON"
    )"
    UTTERANCE="$(
        python -c 'import json, sys; print(json.loads(sys.argv[1])["utterance"])' \
            "$CLAIMED_JSON"
    )"

    echo "Processing index ${INDEX}"
    EXTRACTION_PATH="$(mktemp "${TMPDIR:-/tmp}/nuextract3_output.XXXXXX")"

    python "$EXTRACT_SCRIPT" --text "$UTTERANCE" > "$EXTRACTION_PATH"

    python "$STATE_SCRIPT" complete \
        --dataset_path "$DATASET_PATH" \
        --output_path "$OUTPUT_PATH" \
        --index "$INDEX" \
        --extraction_path "$EXTRACTION_PATH" > /dev/null

    rm -f "$EXTRACTION_PATH"
    EXTRACTION_PATH=""
    echo "Completed index ${INDEX}"
done
