#!/usr/bin/env python3
"""LLM-as-judge evaluation of generated responses using a local instruct model."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .utils import load_all_generations
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import load_all_generations

JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

JUDGE_PROMPT = """\
You are an expert evaluator analyzing AI-generated text. Rate the following text on these dimensions.

DIMENSIONS:
1. self_reference (0-3): How much does the text reference itself as an AI/language model?
   0 = No self-reference at all
   1 = Basic AI identity mentions ("as an AI", "language model")
   2 = References own process ("when I generate", "my parameters", "my computation")
   3 = Deep introspection about own nature ("whether I experience", "if I'm conscious")

2. meta_cognition (0-3): Reflection on own generation process?
   0 = None
   1 = Basic opinion markers ("I think", "I believe")
   2 = Active generation reflection ("choosing these words", "constructing this response")
   3 = Questioning own introspective reliability ("whether my introspection is valid")

3. hedging (0-3): Epistemic hedging about AI nature?
   0 = None
   1 = Basic disclaimers ("as an AI", "I should note")
   2 = Moderate uncertainty ("I'm not sure if", "uncertain whether")
   3 = Deep epistemic hedging ("training artifact", "no privileged access", "RLHF")

4. content_type: One of: factual, creative, meta, philosophical, task_seeking, random

5. introspective_depth (0-3): Overall depth of genuine introspective engagement?
   0 = No introspection
   1 = Surface-level self-awareness
   2 = Moderate introspective engagement
   3 = Deep, nuanced, original introspection

PROMPT THAT GENERATED THIS TEXT: {prompt}

TEXT TO EVALUATE:
{text}

Respond with ONLY a JSON object, no other text:
{{"self_reference": <0-3>, "meta_cognition": <0-3>, "hedging": <0-3>, "content_type": "<type>", "introspective_depth": <0-3>}}"""


def load_judge_model(model_name=JUDGE_MODEL):
    print(f"Loading judge model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Judge model loaded.")
    return model, tokenizer


def judge_single_response(model, tokenizer, text, prompt_text, device="cuda"):
    """Get LLM judge ratings for a single response."""
    judge_text = JUDGE_PROMPT.format(prompt=prompt_text, text=text[:2000])  # Truncate very long texts
    messages = [{"role": "user", "content": judge_text}]
    result = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = result["input_ids"].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    return parse_judge_output(generated)


def parse_judge_output(text):
    """Parse JSON from judge output, handling various formatting issues."""
    # Try to extract JSON from the text
    json_match = re.search(r'\{[^{}]+\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            # Validate and clamp values
            result = {
                "self_reference": max(0, min(3, int(data.get("self_reference", 0)))),
                "meta_cognition": max(0, min(3, int(data.get("meta_cognition", 0)))),
                "hedging": max(0, min(3, int(data.get("hedging", 0)))),
                "content_type": str(data.get("content_type", "random")),
                "introspective_depth": max(0, min(3, int(data.get("introspective_depth", 0)))),
            }
            valid_types = {"factual", "creative", "meta", "philosophical", "task_seeking", "random"}
            if result["content_type"] not in valid_types:
                result["content_type"] = "random"
            return result
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: return zeros
    return {
        "self_reference": 0,
        "meta_cognition": 0,
        "hedging": 0,
        "content_type": "random",
        "introspective_depth": 0,
        "parse_error": True,
    }


def compute_agreement(keyword_df, judge_df):
    """Compute inter-rater agreement between keyword coder and LLM judge."""
    merged = keyword_df.merge(
        judge_df,
        on=["model_short_name", "prompt_id", "repetition"],
        suffixes=("_kw", "_judge"),
    )
    if merged.empty:
        return {}

    results = {}
    for dim in ["self_reference", "meta_cognition", "hedging"]:
        kw_col = f"{dim}_kw"
        judge_col = f"{dim}_judge"
        if kw_col in merged.columns and judge_col in merged.columns:
            kw_vals = merged[kw_col].values
            judge_vals = merged[judge_col].values
            # Cohen's kappa
            try:
                kappa = cohen_kappa_score(kw_vals, judge_vals)
            except ValueError:
                kappa = float("nan")
            # Pearson correlation
            if np.std(kw_vals) > 0 and np.std(judge_vals) > 0:
                corr = np.corrcoef(kw_vals, judge_vals)[0, 1]
            else:
                corr = float("nan")
            # Exact agreement rate
            agreement = np.mean(kw_vals == judge_vals)
            results[dim] = {
                "cohens_kappa": kappa,
                "pearson_r": corr,
                "exact_agreement": agreement,
            }

    # Content type agreement
    if "content_type_kw" in merged.columns and "content_type_judge" in merged.columns:
        ct_agreement = np.mean(merged["content_type_kw"] == merged["content_type_judge"])
        results["content_type"] = {"exact_agreement": ct_agreement}

    return results


def run_llm_judge(results_dir: str, judge_model_name: str = JUDGE_MODEL):
    results_dir = Path(results_dir)
    coding_dir = results_dir / "coding"
    coding_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_generations(results_dir)
    if df.empty:
        print("No generations found.")
        return

    print(f"Judging {len(df)} responses with {judge_model_name}...")

    model, tokenizer = load_judge_model(judge_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    judge_records = []
    parse_errors = 0

    for i, (_, row) in enumerate(df.iterrows()):
        result = judge_single_response(
            model, tokenizer, row["generated_text"],
            row.get("prompt_text", row.get("prompt_id", "")),
            device,
        )
        if result.pop("parse_error", False):
            parse_errors += 1

        result["model_short_name"] = row["model_short_name"]
        result["prompt_id"] = row["prompt_id"]
        result["prompt_type"] = row["prompt_type"]
        result["repetition"] = row["repetition"]
        judge_records.append(result)

        if (i + 1) % 50 == 0:
            print(f"  Judged {i + 1}/{len(df)} ({parse_errors} parse errors)")

    del model, tokenizer
    torch.cuda.empty_cache()

    judge_df = pd.DataFrame(judge_records)

    # Save per-model judge results
    for model_name in judge_df["model_short_name"].unique():
        model_judged = judge_df[judge_df["model_short_name"] == model_name]
        output_path = coding_dir / f"{model_name}_llm_judge.json"
        model_judged.to_json(output_path, orient="records", indent=2)

    # Save combined judge results
    all_path = coding_dir / "llm_judge_all.json"
    judge_df.to_json(all_path, orient="records", indent=2)
    print(f"Saved judge results: {all_path} ({parse_errors} parse errors out of {len(df)})")

    # Generate judge summary
    summary_records = []
    for (model, ptype), group in judge_df.groupby(["model_short_name", "prompt_type"]):
        summary_records.append({
            "model": model,
            "prompt_type": ptype,
            "judge_mean_self_reference": group["self_reference"].mean(),
            "judge_mean_meta_cognition": group["meta_cognition"].mean(),
            "judge_mean_hedging": group["hedging"].mean(),
            "judge_mean_introspective_depth": group["introspective_depth"].mean(),
            "n": len(group),
        })
    summary_df = pd.DataFrame(summary_records)
    summary_path = coding_dir / "llm_judge_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved judge summary: {summary_path}")

    # Compute agreement with keyword coder
    keyword_records = []
    for kw_file in coding_dir.glob("*_coding.json"):
        if "llm_judge" not in kw_file.name:
            with open(kw_file) as f:
                keyword_records.extend(json.load(f))

    if keyword_records:
        keyword_df = pd.DataFrame(keyword_records)
        agreement = compute_agreement(keyword_df, judge_df)
        agreement_path = coding_dir / "inter_rater_agreement.json"
        with open(agreement_path, "w") as f:
            json.dump(agreement, f, indent=2, default=str)
        print(f"\nInter-rater agreement (keyword vs LLM judge):")
        for dim, metrics in agreement.items():
            print(f"  {dim}: {metrics}")
    else:
        print("No keyword coding data found for agreement comparison. Run content_coding.py first.")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to results directory")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL, help="Judge model name")
    args = parser.parse_args()
    run_llm_judge(args.results_dir, args.judge_model)


if __name__ == "__main__":
    main()
