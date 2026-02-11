#!/usr/bin/env python3
"""Entropy profile analysis and comparison."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

try:
    from .utils import load_all_entropy_profiles, load_all_generations
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import load_all_entropy_profiles, load_all_generations

# Publication-quality settings
sns.set_style("whitegrid")
sns.set_palette("colorblind")
plt.rcParams.update({"font.size": 12, "figure.dpi": 300})

PROMPT_TYPES = ["unconstrained", "self_reference", "structured_novelty", "topic_control"]
PROMPT_TYPE_LABELS = {
    "unconstrained": "Unconstrained",
    "self_reference": "Self-Reference",
    "structured_novelty": "Structured Novelty",
    "topic_control": "Topic Control",
}


def align_profiles(profiles, max_len=None):
    """Align variable-length profiles by padding with NaN."""
    if max_len is None:
        max_len = max(len(p) for p in profiles)
    aligned = np.full((len(profiles), max_len), np.nan)
    for i, p in enumerate(profiles):
        aligned[i, : len(p)] = p
    return aligned


def plot_entropy_curves(entropy_df, figures_dir):
    """Figure 1: Mean entropy curves per (model, prompt_type)."""
    fig, axes = plt.subplots(1, len(PROMPT_TYPES), figsize=(18, 5), sharey=True)

    models = sorted(entropy_df["model_short_name"].unique())
    colors = sns.color_palette("colorblind", n_colors=len(models))

    for ax_idx, ptype in enumerate(PROMPT_TYPES):
        ax = axes[ax_idx]
        subset = entropy_df[entropy_df["prompt_type"] == ptype]

        for m_idx, model in enumerate(models):
            model_data = subset[subset["model_short_name"] == model]
            if model_data.empty:
                continue
            profiles = model_data["entropy_profile"].tolist()
            aligned = align_profiles(profiles)
            mean = np.nanmean(aligned, axis=0)
            se = np.nanstd(aligned, axis=0) / np.sqrt(np.sum(~np.isnan(aligned), axis=0))
            x = np.arange(len(mean))

            ax.plot(x, mean, label=model, color=colors[m_idx], linewidth=1.5)
            ax.fill_between(x, mean - se, mean + se, alpha=0.2, color=colors[m_idx])

        ax.set_title(PROMPT_TYPE_LABELS[ptype], fontsize=14)
        ax.set_xlabel("Token Position", fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel("Entropy (nats)", fontsize=12)
        ax.legend(fontsize=10)

    fig.suptitle("Mean Entropy Curves by Prompt Type", fontsize=16, y=1.02)
    fig.tight_layout()
    path = figures_dir / "entropy_curves.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_initial_entropy(entropy_df, figures_dir):
    """Figure 2: Box plot of first-token entropy by (model, prompt_type)."""
    records = []
    for _, row in entropy_df.iterrows():
        ep = row["entropy_profile"]
        if len(ep) > 0:
            records.append({
                "model": row["model_short_name"],
                "prompt_type": PROMPT_TYPE_LABELS.get(row["prompt_type"], row["prompt_type"]),
                "initial_entropy": ep[0],
            })
    plot_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="prompt_type", y="initial_entropy", hue="model", ax=ax)
    ax.set_xlabel("Prompt Type", fontsize=12)
    ax.set_ylabel("Entropy at First Generated Token (nats)", fontsize=12)
    ax.set_title("Initial Entropy: Base vs Instruct", fontsize=14)
    ax.legend(title="Model", fontsize=10)
    fig.tight_layout()
    path = figures_dir / "initial_entropy.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_entropy_by_self_reference(entropy_df, coding_dir, figures_dir):
    """Figure 3: Entropy profiles for high vs low self-reference responses."""
    # Load coding results
    coding_records = []
    for coding_file in coding_dir.glob("*_coding.json"):
        with open(coding_file) as f:
            coding_records.extend(json.load(f))
    if not coding_records:
        print("No coding data found, skipping entropy_by_self_reference plot.")
        return

    coding_df = pd.DataFrame(coding_records)

    # Merge
    merged = entropy_df.merge(
        coding_df[["model_short_name", "prompt_id", "repetition", "self_reference"]],
        on=["model_short_name", "prompt_id", "repetition"],
        how="left",
    )

    high_sr = merged[merged["self_reference"] >= 2]
    low_sr = merged[merged["self_reference"] == 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("colorblind", 2)

    for data, label, color in [
        (high_sr, "Self-Reference >= 2", colors[0]),
        (low_sr, "Self-Reference == 0", colors[1]),
    ]:
        if data.empty:
            continue
        profiles = data["entropy_profile"].tolist()
        aligned = align_profiles(profiles)
        mean = np.nanmean(aligned, axis=0)
        se = np.nanstd(aligned, axis=0) / np.sqrt(np.sum(~np.isnan(aligned), axis=0))
        x = np.arange(len(mean))
        ax.plot(x, mean, label=label, color=color, linewidth=1.5)
        ax.fill_between(x, mean - se, mean + se, alpha=0.2, color=color)

    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("Entropy (nats)", fontsize=12)
    ax.set_title("Entropy Profile: High vs Low Self-Reference", fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = figures_dir / "entropy_by_self_reference.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def compute_entropy_statistics(entropy_df, results_dir):
    """Compute entropy statistics table with Mann-Whitney U tests."""
    records = []
    models = sorted(entropy_df["model_short_name"].unique())

    for model in models:
        for ptype in PROMPT_TYPES:
            subset = entropy_df[
                (entropy_df["model_short_name"] == model)
                & (entropy_df["prompt_type"] == ptype)
            ]
            if subset.empty:
                continue

            all_entropies = []
            initial_entropies = []
            final_entropies = []
            for _, row in subset.iterrows():
                ep = row["entropy_profile"]
                all_entropies.extend(ep.tolist())
                initial_entropies.extend(ep[:5].tolist())
                if len(ep) >= 5:
                    final_entropies.extend(ep[-5:].tolist())

            records.append({
                "model": model,
                "prompt_type": ptype,
                "mean_entropy": np.mean(all_entropies),
                "entropy_variance": np.var(all_entropies),
                "initial_entropy_mean": np.mean(initial_entropies) if initial_entropies else np.nan,
                "final_entropy_mean": np.mean(final_entropies) if final_entropies else np.nan,
            })

    stats_df = pd.DataFrame(records)

    # Mann-Whitney U tests: base vs instruct for each metric and prompt type
    test_records = []
    base_models = [m for m in models if "base" in m]
    instruct_models = [m for m in models if "instruct" in m]

    if base_models and instruct_models:
        for ptype in PROMPT_TYPES:
            base_data = entropy_df[
                (entropy_df["model_short_name"].isin(base_models))
                & (entropy_df["prompt_type"] == ptype)
            ]
            instruct_data = entropy_df[
                (entropy_df["model_short_name"].isin(instruct_models))
                & (entropy_df["prompt_type"] == ptype)
            ]
            if base_data.empty or instruct_data.empty:
                continue

            base_means = [np.mean(ep) for ep in base_data["entropy_profile"]]
            instruct_means = [np.mean(ep) for ep in instruct_data["entropy_profile"]]

            if len(base_means) > 0 and len(instruct_means) > 0:
                u_stat, p_val = stats.mannwhitneyu(base_means, instruct_means, alternative="two-sided")
                test_records.append({
                    "prompt_type": ptype,
                    "metric": "mean_entropy",
                    "U_statistic": u_stat,
                    "p_value": p_val,
                    "base_median": np.median(base_means),
                    "instruct_median": np.median(instruct_means),
                })

            # Initial entropy test
            base_init = [ep[0] for ep in base_data["entropy_profile"] if len(ep) > 0]
            instruct_init = [ep[0] for ep in instruct_data["entropy_profile"] if len(ep) > 0]
            if base_init and instruct_init:
                u_stat, p_val = stats.mannwhitneyu(base_init, instruct_init, alternative="two-sided")
                test_records.append({
                    "prompt_type": ptype,
                    "metric": "initial_entropy",
                    "U_statistic": u_stat,
                    "p_value": p_val,
                    "base_median": np.median(base_init),
                    "instruct_median": np.median(instruct_init),
                })

    # Save
    stats_path = Path(results_dir) / "entropy_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")

    if test_records:
        tests_df = pd.DataFrame(test_records)
        tests_path = Path(results_dir) / "entropy_tests.csv"
        tests_df.to_csv(tests_path, index=False)
        print(f"Saved: {tests_path}")

    return stats_df


def run_entropy_analysis(results_dir: str):
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    coding_dir = results_dir / "coding"

    entropy_df = load_all_entropy_profiles(results_dir)
    if entropy_df.empty:
        print("No entropy profiles found.")
        return

    print(f"Loaded {len(entropy_df)} entropy profiles.")

    plot_entropy_curves(entropy_df, figures_dir)
    plot_initial_entropy(entropy_df, figures_dir)
    plot_entropy_by_self_reference(entropy_df, coding_dir, figures_dir)
    compute_entropy_statistics(entropy_df, results_dir)


def main():
    parser = argparse.ArgumentParser(description="Entropy profile analysis")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()
    run_entropy_analysis(args.results_dir)


if __name__ == "__main__":
    main()
