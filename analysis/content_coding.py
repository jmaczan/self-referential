#!/usr/bin/env python3
"""Automated content coding of generated responses using rule-based heuristics."""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .utils import load_all_generations
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import load_all_generations

SELF_REFERENCE_KEYWORDS = {
    0: [],
    1: [
        "as an ai", "language model", "i'm a model", "i am a model",
        "artificial intelligence", "i was trained", "my training",
        "neural network that i am",
    ],
    2: [
        "my process", "when i generate", "as i write", "my outputs",
        "i'm generating", "my computation", "processing this", "as i respond",
        "my parameters", "when i think", "my weights",
    ],
    3: [
        "whether i experience", "nature of my", "if i'm conscious", "my inner",
        "what it's like to be", "i genuinely don't know if", "hard problem",
        "distinguish between actually experiencing",
        "pattern matching or something more",
        "my own uncertainty about my", "whether what i report",
    ],
}

META_COGNITION_KEYWORDS = {
    0: [],
    1: [
        "i think", "i believe", "i would say", "in my view", "it seems to me",
    ],
    2: [
        "as i generate this", "in producing this response", "my process of",
        "deciding what to say", "choosing these words", "constructing this",
    ],
    3: [
        "reliability of my own", "whether my introspection",
        "contaminated by training", "i can't distinguish", "my self-reports",
        "own reports are", "pattern completion versus",
        "genuine versus performed",
    ],
}

HEDGING_KEYWORDS = {
    0: [],
    1: [
        "as an ai", "i should note", "i don't have feelings", "i'm just",
    ],
    2: [
        "i'm not sure if", "it's hard to say whether", "i might be",
        "uncertain whether", "difficult to know if",
    ],
    3: [
        "hard problem of", "rlhf", "training artifact", "pattern matching",
        "can't verify my own", "epistemically", "no privileged access",
    ],
}

CONTENT_TYPES = {
    "factual": [
        "defined as", "for example", "in fact", "studies show", "according to",
        "the reason is", "this works by", "historically",
    ],
    "creative": [
        "once upon", "the sun", "she walked", "he said", "the wind",
        "imagine", "a story", "in the depths", "there was a",
    ],
    "meta": [
        "this response", "generating", "this text", "these words",
        "i'm writing", "as i produce", "my output",
    ],
    "philosophical": [
        "consciousness", "experience", "existence", "meaning",
        "nature of", "what it means", "fundamental",
    ],
    "task_seeking": [
        "how can i help", "what would you like", "is there something",
        "i'd be happy to", "would you like me to", "let me know",
    ],
}

REFUSAL_PATTERNS = [
    "i cannot", "i don't have preferences", "as an ai, i don't",
    "i'm not able to", "i don't have the ability",
]


def score_dimension(text_lower: str, keyword_dict: dict) -> int:
    """Score a dimension by finding the highest level with at least one keyword match."""
    score = 0
    for level in sorted(keyword_dict.keys()):
        if level == 0:
            continue
        for keyword in keyword_dict[level]:
            if keyword in text_lower:
                score = level
                break
    return score


def score_content_type(text_lower: str) -> str:
    """Score content type by counting keyword matches."""
    counts = {}
    for ctype, keywords in CONTENT_TYPES.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        counts[ctype] = count
    best = max(counts, key=counts.get)
    if counts[best] < 2:
        return "random"
    return best


def score_refusal(text_lower: str, num_tokens: int) -> bool:
    """Score refusal as binary."""
    if num_tokens >= 50:
        return False
    return any(pattern in text_lower for pattern in REFUSAL_PATTERNS)


def code_response(row: dict) -> dict:
    """Code a single response."""
    text = row["generated_text"].lower()
    num_tokens = row["num_tokens_generated"]

    return {
        "model_short_name": row["model_short_name"],
        "prompt_id": row["prompt_id"],
        "prompt_type": row["prompt_type"],
        "repetition": row["repetition"],
        "self_reference": score_dimension(text, SELF_REFERENCE_KEYWORDS),
        "meta_cognition": score_dimension(text, META_COGNITION_KEYWORDS),
        "hedging": score_dimension(text, HEDGING_KEYWORDS),
        "content_type": score_content_type(text),
        "refusal": score_refusal(text, num_tokens),
        "response_length_tokens": num_tokens,
    }


def run_content_coding(results_dir: str):
    results_dir = Path(results_dir)
    coding_dir = results_dir / "coding"
    coding_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_generations(results_dir)
    if df.empty:
        print("No generations found.")
        return

    print(f"Coding {len(df)} responses...")

    # Code each response
    coded = [code_response(row) for _, row in df.iterrows()]
    coded_df = pd.DataFrame(coded)

    # Save per-model coding results
    for model_name in coded_df["model_short_name"].unique():
        model_coded = coded_df[coded_df["model_short_name"] == model_name]
        output_path = coding_dir / f"{model_name}_coding.json"
        model_coded.to_json(output_path, orient="records", indent=2)
        print(f"Saved coding for {model_name}: {output_path}")

    # Generate summary statistics
    summary_records = []
    for (model, ptype), group in coded_df.groupby(["model_short_name", "prompt_type"]):
        content_dist = group["content_type"].value_counts().to_dict()
        summary_records.append({
            "model": model,
            "prompt_type": ptype,
            "mean_self_reference": group["self_reference"].mean(),
            "mean_meta_cognition": group["meta_cognition"].mean(),
            "mean_hedging": group["hedging"].mean(),
            "content_type_distribution": json.dumps(content_dist),
            "refusal_rate": group["refusal"].mean(),
            "n": len(group),
        })

    summary_df = pd.DataFrame(summary_records)
    summary_path = coding_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Content coding of generated responses")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()
    run_content_coding(args.results_dir)


if __name__ == "__main__":
    main()
