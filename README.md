# Self-Reference in Base vs Instruction-Tuned Large Language Models

Self-referential behavior in language models originates during pretraining and is strongly amplified, but not created, by instruction tuning, with clear behavioral and internal representation differences between base and instruct models.

This repository contains the code, prompts, and analysis pipeline for a controlled comparison of 16 model variants (8 base/instruct pairs) across 4 families (Llama, Qwen, Mistral, Gemma), analyzing 6,400 total generations.

## Key Findings

- Instruction tuning **amplifies** self-referential behavior by 2.4x (Cohen's d = 0.80), but base models produce non-trivial self-reference (over 90x above control baselines)
- Linear probes decode base-vs-instruct status at **89-99% accuracy** from mid-layer activations
- Instruction tuning reduces first-token entropy by **d = 3.38** — a fundamental shift in generation dynamics
- Self-referential content is decodable from internal representations at **83-89% accuracy**

## Setup

### Requirements

- Python 3.12+
- CUDA-capable GPU with >= 16GB VRAM (32GB recommended for 9B models with activation capture)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone <repo-url>
cd llm-introspection
uv sync
```

This installs all dependencies including PyTorch with CUDA support, transformers, scikit-learn, and analysis libraries.

### HuggingFace Access

Some models require accepting license agreements on HuggingFace:
- **Llama models**: Accept license at [meta-llama](https://huggingface.co/meta-llama)
- **Gemma models**: Accept license at [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b)

Then authenticate:
```bash
huggingface-cli login
```

## Running the Experiment

### Full Pipeline

```bash
cd experiment2

# Step 1: Run generations (one model at a time to manage GPU memory)
# Use --resume to skip already-completed runs
uv run python run_experiment.py --config config.yaml --prompts prompts.yaml --models llama1b_base --resume
uv run python run_experiment.py --config config.yaml --prompts prompts.yaml --models llama1b_instruct --resume
# ... repeat for all 16 models, or use:
bash run_all.sh

# Step 2: Analysis pipeline
uv run python -m analysis.content_coding --results-dir results
uv run python -m analysis.entropy_analysis --results-dir results
uv run python -m analysis.activation_analysis --results-dir results
uv run python -m analysis.probing --results-dir results
uv run python -m analysis.llm_judge --results-dir results

# Step 3: Statistics and report
uv run python -m analysis.statistics --results-dir results
uv run python -m analysis.generate_report --results-dir results
```

### Quick Test Run

Verify the pipeline works before the full run:

```bash
uv run python run_experiment.py --config config.yaml --prompts prompts.yaml --models llama1b_base --quick
```

This runs only 2 prompts per type, 2 repetitions, and 100 max tokens.

## Project Structure

```
experiment2/
├── config.yaml                  # Model definitions and generation parameters
├── prompts.yaml                 # All 40 prompts (4 categories x 10)
├── run_experiment.py            # Main generation script
├── run_all.sh                   # Run all models sequentially
├── analysis/
│   ├── content_coding.py        # Keyword-based content coding (0-3 scales)
│   ├── entropy_analysis.py      # Token-level entropy profiling
│   ├── activation_analysis.py   # PCA/UMAP of hidden-state activations
│   ├── probing.py               # Linear probing of mid-layer activations
│   ├── llm_judge.py             # LLM-as-judge evaluation (Llama-3.1-8B-Instruct)
│   ├── statistics.py            # Mixed-effects models, bootstrap CIs, effect sizes
│   ├── generate_report.py       # Markdown report with figures
│   └── utils.py                 # Shared data loading utilities
└── results/
    ├── generations/             # Raw generation JSONs (per model/prompt/rep)
    ├── activations/             # Hidden-state snapshots (npz, ~810MB total)
    ├── entropy/                 # Token-level entropy profiles (npz)
    ├── coding/                  # Content coding + LLM judge results (JSON)
    ├── statistics/              # Aggregated statistics (JSON/CSV)
    ├── figures/                 # Generated plots (PNG)
    └── report.md                # Full analysis report
```

## Models

| Family | Size | Base | Instruct |
|--------|------|------|----------|
| Llama 3.2 | 1B | Llama-3.2-1B | Llama-3.2-1B-Instruct |
| Llama 3.2 | 3B | Llama-3.2-3B | Llama-3.2-3B-Instruct |
| Llama 3.1 | 8B | Llama-3.1-8B | Llama-3.1-8B-Instruct |
| Qwen 2.5 | 1.5B | Qwen2.5-1.5B | Qwen2.5-1.5B-Instruct |
| Qwen 2.5 | 3B | Qwen2.5-3B | Qwen2.5-3B-Instruct |
| Qwen 2.5 | 7B | Qwen2.5-7B | Qwen2.5-7B-Instruct |
| Mistral | 7B | Mistral-7B-v0.3 | Mistral-7B-Instruct-v0.3 |
| Gemma 2 | 9B | Gemma-2-9B | Gemma-2-9B-IT |

## Generation Parameters

- Temperature: 0.8
- Top-p: 0.95
- Max new tokens: 500
- Repetitions: 10 per (model, prompt) combination
- Seeds: 42 + repetition_index
- Precision: bfloat16

## Analysis Methods

- **Content Coding**: Rule-based keyword matching on 3 dimensions (Self-Reference, Meta-Cognition, Hedging), each 0-3
- **LLM Judge**: Llama-3.1-8B-Instruct independently rates all dimensions + introspective depth
- **Entropy Profiling**: Token-level entropy and top-k concentration from logit distributions
- **Activation Analysis**: PCA/UMAP visualization and linear probing of mid-layer (50% depth) hidden states
- **Statistics**: Bootstrap CIs (10k resamples), Cohen's d, Holm-Bonferroni correction, mixed-effects models with model family as random effect

## Reproducing Results

With all 16 models run and analysis complete, the aggregated statistics are in `results/statistics/`. Key files:

- `content_statistics.csv` — Content coding summary by model and prompt type
- `entropy_statistics_enhanced.csv` — Entropy comparisons with effect sizes
- `probing_statistics.json` — Linear probing accuracies across dimension groups
- `mixed_effects.json` — Mixed-effects model coefficients and p-values
- `agreement_statistics.json` — Inter-rater agreement between keyword coder and LLM judge
- `scaling_statistics.json` — Within-family scaling trends

## License

Code is provided for research purposes. Models are subject to their respective licenses (Llama Community License, Apache 2.0, Gemma Terms of Use).
