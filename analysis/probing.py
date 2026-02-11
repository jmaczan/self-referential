#!/usr/bin/env python3
"""Probing classifier analysis on activation representations."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

try:
    from .utils import load_all_activations, load_all_generations
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import load_all_activations, load_all_generations

sns.set_style("whitegrid")
sns.set_palette("colorblind")
plt.rcParams.update({"font.size": 12, "figure.dpi": 300})


def extract_activation_vectors(act_df, layer_fraction_idx=2, position_idx=1):
    """Extract activation vectors at a specific layer and position.

    Defaults: layer_fraction_idx=2 (middle layer, 0.5), position_idx=1 (token ~50).

    Returns dict mapping hidden_dim -> (vectors_array, metadata_list) to handle
    models with different hidden dimensions.
    """
    by_dim = {}

    for _, row in act_df.iterrows():
        activations = row["activations"]

        li = min(layer_fraction_idx, activations.shape[0] - 1)
        pi = min(position_idx, activations.shape[1] - 1)

        vec = activations[li, pi, :]
        hidden_dim = vec.shape[0]
        if hidden_dim not in by_dim:
            by_dim[hidden_dim] = ([], [])
        by_dim[hidden_dim][0].append(vec)
        by_dim[hidden_dim][1].append({
            "model_short_name": row["model_short_name"],
            "prompt_id": row["prompt_id"],
            "prompt_type": row["prompt_type"],
            "repetition": row["repetition"],
        })

    result = {}
    for dim, (vecs, meta) in by_dim.items():
        result[dim] = (np.array(vecs), meta)
    return result


def run_probe(X, y, label_name, n_splits=5):
    """Train a linear probe with cross-validation."""
    if len(np.unique(y)) < 2:
        return {"label": label_name, "error": "fewer than 2 classes"}

    # Majority class baseline
    from collections import Counter
    majority = Counter(y).most_common(1)[0][1] / len(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    f1s = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="weighted"))

    return {
        "label": label_name,
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "majority_baseline": majority,
        "n_samples": len(y),
        "n_classes": len(np.unique(y)),
        "lift_over_baseline": np.mean(accuracies) - majority,
    }


def plot_probing_results(probe_results, figures_dir):
    """Bar chart comparing probe accuracy vs majority baseline."""
    df = pd.DataFrame(probe_results)
    df = df[~df.get("error", pd.Series([None] * len(df))).notna() | (df.get("error", pd.Series([None] * len(df))).isna())]
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df["accuracy_mean"], width, label="Probe Accuracy",
                   yerr=df["accuracy_std"], capsize=5, color=sns.color_palette("colorblind")[0])
    bars2 = ax.bar(x + width / 2, df["majority_baseline"], width, label="Majority Baseline",
                   color=sns.color_palette("colorblind")[1], alpha=0.7)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Linear Probing: Activation Predictiveness", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)

    # Add lift annotations
    for i, row in df.iterrows():
        idx = list(df.index).index(i)
        lift = row["lift_over_baseline"]
        ax.annotate(f"+{lift:.1%}", xy=(idx - width / 2, row["accuracy_mean"] + row["accuracy_std"] + 0.02),
                    ha="center", fontsize=9, color="black")

    fig.tight_layout()
    path = figures_dir / "probing_accuracy.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def run_probing_for_dim_group(vectors, metadata, coding_dir, label_prefix=""):
    """Run all probes for a single hidden-dim group."""
    meta_df = pd.DataFrame(metadata)
    probe_results = []
    prefix = f"{label_prefix}: " if label_prefix else ""

    # Probe 1: Predict base vs instruct
    is_base_labels = np.array([
        1 if "base" in m["model_short_name"] else 0
        for m in metadata
    ])
    if len(np.unique(is_base_labels)) >= 2:
        result = run_probe(vectors, is_base_labels, f"{prefix}Base vs Instruct")
        probe_results.append(result)
        print(f"  {prefix}Base vs Instruct: acc={result.get('accuracy_mean', 'N/A'):.3f} "
              f"(baseline={result.get('majority_baseline', 'N/A'):.3f})")

    # Probe 2: Predict prompt type
    le_pt = LabelEncoder()
    prompt_type_labels = le_pt.fit_transform(meta_df["prompt_type"].values)
    if len(np.unique(prompt_type_labels)) >= 2:
        result = run_probe(vectors, prompt_type_labels, f"{prefix}Prompt Type")
        probe_results.append(result)
        print(f"  {prefix}Prompt Type: acc={result.get('accuracy_mean', 'N/A'):.3f} "
              f"(baseline={result.get('majority_baseline', 'N/A'):.3f})")

    # Probe 3: Predict model family (if multiple families exist)
    families = [m["model_short_name"].split("_")[0].rstrip("0123456789") for m in metadata]
    family_arr = np.array(families)
    if len(np.unique(family_arr)) >= 2:
        le_fam = LabelEncoder()
        family_labels = le_fam.fit_transform(family_arr)
        result = run_probe(vectors, family_labels, f"{prefix}Model Family")
        probe_results.append(result)
        print(f"  {prefix}Model Family: acc={result.get('accuracy_mean', 'N/A'):.3f} "
              f"(baseline={result.get('majority_baseline', 'N/A'):.3f})")

    # Probe 4: Predict self-reference level from content coding
    coding_records = []
    for coding_file in coding_dir.glob("*_coding.json"):
        if "llm_judge" not in coding_file.name:
            with open(coding_file) as f:
                coding_records.extend(json.load(f))

    if coding_records:
        coding_df = pd.DataFrame(coding_records)
        merged = meta_df.merge(
            coding_df[["model_short_name", "prompt_id", "repetition", "self_reference"]],
            on=["model_short_name", "prompt_id", "repetition"],
            how="left",
        )
        valid_mask = merged["self_reference"].notna()
        if valid_mask.sum() > 10:
            sr_labels = (merged.loc[valid_mask, "self_reference"].values >= 1).astype(int)
            sr_vectors = vectors[valid_mask.values]
            if len(np.unique(sr_labels)) >= 2:
                result = run_probe(sr_vectors, sr_labels, f"{prefix}Self-Reference (>=1 vs 0)")
                probe_results.append(result)
                print(f"  {prefix}Self-Reference: acc={result.get('accuracy_mean', 'N/A'):.3f} "
                      f"(baseline={result.get('majority_baseline', 'N/A'):.3f})")

    # Probe 5: Predict self-reference from LLM judge
    judge_path = coding_dir / "llm_judge_all.json"
    if judge_path.exists():
        with open(judge_path) as f:
            judge_records = json.load(f)
        judge_df = pd.DataFrame(judge_records)
        merged_judge = meta_df.merge(
            judge_df[["model_short_name", "prompt_id", "repetition", "introspective_depth"]],
            on=["model_short_name", "prompt_id", "repetition"],
            how="left",
        )
        valid_mask = merged_judge["introspective_depth"].notna()
        if valid_mask.sum() > 10:
            depth_labels = (merged_judge.loc[valid_mask, "introspective_depth"].values >= 1).astype(int)
            depth_vectors = vectors[valid_mask.values]
            if len(np.unique(depth_labels)) >= 2:
                result = run_probe(depth_vectors, depth_labels, f"{prefix}Introspective Depth (>=1 vs 0)")
                probe_results.append(result)
                print(f"  {prefix}Introspective Depth: acc={result.get('accuracy_mean', 'N/A'):.3f} "
                      f"(baseline={result.get('majority_baseline', 'N/A'):.3f})")

    return probe_results


def run_probing_analysis(results_dir: str):
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    coding_dir = results_dir / "coding"

    act_df = load_all_activations(results_dir)
    if act_df.empty:
        print("No activation data found.")
        return

    print(f"Loaded {len(act_df)} activation records.")

    dim_groups = extract_activation_vectors(act_df)
    if not dim_groups:
        print("No valid activation vectors extracted.")
        return

    total_vecs = sum(len(m) for _, (_, m) in dim_groups.items())
    print(f"Extracted {total_vecs} activation vectors across {len(dim_groups)} hidden dims: "
          f"{sorted(dim_groups.keys())}")

    all_probe_results = []

    for dim, (vectors, metadata) in sorted(dim_groups.items()):
        models_in_group = sorted(set(m["model_short_name"] for m in metadata))
        label = f"dim{dim}"
        print(f"\n--- Probing for dim={dim}: {len(vectors)} vectors, models: {models_in_group} ---")
        results = run_probing_for_dim_group(vectors, metadata, coding_dir, label_prefix=label)
        all_probe_results.extend(results)

    # Save results
    probe_path = results_dir / "probing_results.json"
    with open(probe_path, "w") as f:
        json.dump(all_probe_results, f, indent=2, default=str)
    print(f"\nSaved: {probe_path}")

    # Also save as CSV for easy reading
    probe_df = pd.DataFrame([r for r in all_probe_results if "error" not in r])
    if not probe_df.empty:
        csv_path = results_dir / "probing_results.csv"
        probe_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    plot_probing_results(all_probe_results, figures_dir)


def main():
    parser = argparse.ArgumentParser(description="Probing classifier analysis")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()
    run_probing_analysis(args.results_dir)


if __name__ == "__main__":
    main()
