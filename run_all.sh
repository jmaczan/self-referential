#!/bin/bash
# Run all experiments sequentially, one model at a time
# Uses --resume to skip already-completed runs
set -e

cd "$(dirname "$0")"

MODELS=(
    # Small models first (fastest)
    "llama1b_base"
    "llama1b_instruct"
    "qwen1_5b_base"
    "qwen1_5b_instruct"
    # Medium models
    "llama3b_base"
    "llama3b_instruct"
    "qwen3b_base"
    "qwen3b_instruct"
    # Large models (already have llama8b, run with --resume for control prompts)
    "llama8b_base"
    "llama8b_instruct"
    "mistral7b_base"
    "mistral7b_instruct"
    "qwen7b_base"
    "qwen7b_instruct"
    "gemma9b_base"
    "gemma9b_it"
)

for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Running: $model"
    echo "=========================================="
    uv run python run_experiment.py --config config.yaml --prompts prompts.yaml --models "$model" --resume
    echo ""
done

echo "All experiments complete."

# Run analysis pipeline
echo "=========================================="
echo "Running analysis pipeline..."
echo "=========================================="
uv run python -m analysis.content_coding --results-dir results
uv run python -m analysis.entropy_analysis --results-dir results
uv run python -m analysis.activation_analysis --results-dir results
uv run python -m analysis.probing --results-dir results
uv run python -m analysis.llm_judge --results-dir results
uv run python -m analysis.generate_report --results-dir results
echo "Done."
