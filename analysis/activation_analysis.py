#!/usr/bin/env python3
"""Activation geometry analysis (PCA/UMAP)."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

try:
    from .utils import load_all_activations
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import load_all_activations

sns.set_style("whitegrid")
sns.set_palette("colorblind")
plt.rcParams.update({"font.size": 12, "figure.dpi": 300})

PROMPT_TYPE_LABELS = {
    "unconstrained": "Unconstrained",
    "self_reference": "Self-Reference",
    "structured_novelty": "Structured Novelty",
    "topic_control": "Topic Control",
}

# Dynamic: assign markers based on base/instruct status
def get_model_marker(model_name):
    if "instruct" in model_name or model_name.endswith("_it"):
        return "^"
    return "o"


def extract_mid_layer_activations(act_df):
    """Extract activations from the middle layer at the nearest position to token 50.

    Returns a dict mapping hidden_dim -> (vectors_array, metadata_list) to handle
    models with different hidden dimensions.
    """
    by_dim = {}  # hidden_dim -> (vectors_list, metadata_list)

    for _, row in act_df.iterrows():
        activations = row["activations"]  # (num_layers, num_positions, hidden_dim)
        layer_indices = row["layer_indices"]
        token_positions = row["token_positions"]

        # Find middle layer (layer_fraction=0.5 -> index 2 in default config)
        mid_idx = len(layer_indices) // 2

        # Find position closest to 50 (or nearest available)
        if len(token_positions) == 0:
            continue
        target = 49
        pos_idx = np.argmin(np.abs(token_positions - target))

        if mid_idx < activations.shape[0] and pos_idx < activations.shape[1]:
            vec = activations[mid_idx, pos_idx, :]
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


def plot_pca_by_type(vectors, metadata, figures_dir, suffix=""):
    """Figure 4: PCA colored by prompt_type, marker by model."""
    if len(vectors) < 3:
        print("Not enough activation vectors for PCA.")
        return

    pca = PCA(n_components=2)
    projected = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(10, 8))

    type_colors = {
        "unconstrained": sns.color_palette("colorblind")[0],
        "self_reference": sns.color_palette("colorblind")[1],
        "structured_novelty": sns.color_palette("colorblind")[2],
        "topic_control": sns.color_palette("colorblind")[3],
    }

    for i, meta in enumerate(metadata):
        ptype = meta["prompt_type"]
        model = meta["model_short_name"]
        ax.scatter(
            projected[i, 0],
            projected[i, 1],
            c=[type_colors.get(ptype, "gray")],
            marker=get_model_marker(model),
            s=60,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )

    from matplotlib.lines import Line2D
    type_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=type_colors.get(t, "gray"),
               markersize=10, label=PROMPT_TYPE_LABELS.get(t, t))
        for t in sorted(set(m["prompt_type"] for m in metadata)) if t in type_colors
    ]
    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=10, label="Base"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markersize=10, label="Instruct"),
    ]
    ax.legend(handles=type_handles + model_handles, fontsize=10, loc="best")

    models_str = ", ".join(sorted(set(m["model_short_name"] for m in metadata)))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=12)
    title = "Activation PCA: Colored by Prompt Type"
    if suffix:
        title += f" ({models_str})"
    ax.set_title(title, fontsize=12 if suffix else 14)
    fig.tight_layout()
    path = figures_dir / f"activation_pca_by_type{suffix}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    return pca, projected


def plot_umap_by_type(vectors, metadata, figures_dir, suffix=""):
    """UMAP version of Figure 4."""
    if len(vectors) < 15:
        print("Not enough activation vectors for UMAP.")
        return
    try:
        import umap
    except ImportError:
        print("umap-learn not installed, skipping UMAP plot.")
        return

    reducer = umap.UMAP(n_components=2, random_state=42)
    projected = reducer.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(10, 8))

    type_colors = {
        "unconstrained": sns.color_palette("colorblind")[0],
        "self_reference": sns.color_palette("colorblind")[1],
        "structured_novelty": sns.color_palette("colorblind")[2],
        "topic_control": sns.color_palette("colorblind")[3],
    }

    for i, meta in enumerate(metadata):
        ptype = meta["prompt_type"]
        model = meta["model_short_name"]
        ax.scatter(
            projected[i, 0],
            projected[i, 1],
            c=[type_colors.get(ptype, "gray")],
            marker=get_model_marker(model),
            s=60,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )

    from matplotlib.lines import Line2D
    type_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=type_colors.get(t, "gray"),
               markersize=10, label=PROMPT_TYPE_LABELS.get(t, t))
        for t in sorted(set(m["prompt_type"] for m in metadata)) if t in type_colors
    ]
    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=10, label="Base"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markersize=10, label="Instruct"),
    ]
    ax.legend(handles=type_handles + model_handles, fontsize=10, loc="best")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    title = "Activation UMAP: Colored by Prompt Type"
    if suffix:
        models_str = ", ".join(sorted(set(m["model_short_name"] for m in metadata)))
        title += f" ({models_str})"
    ax.set_title(title, fontsize=12 if suffix else 14)
    fig.tight_layout()
    path = figures_dir / f"activation_umap_by_type{suffix}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pca_by_model(vectors, metadata, figures_dir, suffix=""):
    """Figure 5: PCA colored by base vs instruct."""
    if len(vectors) < 3:
        print("Not enough activation vectors for PCA by model.")
        return

    pca = PCA(n_components=2)
    projected = pca.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = sns.color_palette("colorblind", 2)
    base_color, instruct_color = colors[0], colors[1]

    for i, meta in enumerate(metadata):
        model = meta["model_short_name"]
        is_instruct = "instruct" in model or model.endswith("_it")
        c = instruct_color if is_instruct else base_color
        ax.scatter(
            projected[i, 0],
            projected[i, 1],
            c=[c],
            marker="^" if is_instruct else "o",
            s=60,
            alpha=0.5,
            edgecolors="white",
            linewidth=0.5,
        )

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=base_color,
               markersize=10, label="Base models"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=instruct_color,
               markersize=10, label="Instruct models"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="best")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=12)
    title = "Activation PCA: Base vs Instruct"
    if suffix:
        models_str = ", ".join(sorted(set(m["model_short_name"] for m in metadata)))
        title += f" ({models_str})"
    ax.set_title(title, fontsize=12 if suffix else 14)
    fig.tight_layout()
    path = figures_dir / f"activation_pca_by_model{suffix}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def compute_activation_distances(vectors, metadata, results_dir, suffix=""):
    """Compute pairwise cosine distances between groups. Returns records list."""
    if len(vectors) < 2:
        print("Not enough vectors for distance analysis.")
        return []

    meta_df = pd.DataFrame(metadata)
    dist_matrix = cosine_distances(vectors)

    records = []

    # Intra-group: same (model, prompt_type)
    for (model, ptype), group in meta_df.groupby(["model_short_name", "prompt_type"]):
        idxs = group.index.tolist()
        if len(idxs) < 2:
            continue
        dists = [dist_matrix[i, j] for i in idxs for j in idxs if i < j]
        records.append({
            "comparison": f"intra_{model}_{ptype}",
            "type": "intra_group",
            "model": model,
            "prompt_type": ptype,
            "mean_cosine_distance": np.mean(dists),
            "std_cosine_distance": np.std(dists),
            "n_pairs": len(dists),
        })

    # Inter-type: same model, different prompt_type
    for model, model_group in meta_df.groupby("model_short_name"):
        ptypes = model_group["prompt_type"].unique()
        for i_t, pt1 in enumerate(ptypes):
            for pt2 in ptypes[i_t + 1:]:
                idxs1 = model_group[model_group["prompt_type"] == pt1].index.tolist()
                idxs2 = model_group[model_group["prompt_type"] == pt2].index.tolist()
                dists = [dist_matrix[i, j] for i in idxs1 for j in idxs2]
                if dists:
                    records.append({
                        "comparison": f"inter_type_{model}_{pt1}_vs_{pt2}",
                        "type": "inter_type",
                        "model": model,
                        "prompt_type": f"{pt1} vs {pt2}",
                        "mean_cosine_distance": np.mean(dists),
                        "std_cosine_distance": np.std(dists),
                        "n_pairs": len(dists),
                    })

    # Inter-model: same prompt_type, different model (only within same dim group)
    models = meta_df["model_short_name"].unique()
    if len(models) >= 2:
        for ptype in meta_df["prompt_type"].unique():
            for i_m, m1 in enumerate(models):
                for m2 in models[i_m + 1:]:
                    idxs1 = meta_df[
                        (meta_df["model_short_name"] == m1) & (meta_df["prompt_type"] == ptype)
                    ].index.tolist()
                    idxs2 = meta_df[
                        (meta_df["model_short_name"] == m2) & (meta_df["prompt_type"] == ptype)
                    ].index.tolist()
                    dists = [dist_matrix[i, j] for i in idxs1 for j in idxs2]
                    if dists:
                        records.append({
                            "comparison": f"inter_model_{ptype}_{m1}_vs_{m2}",
                            "type": "inter_model",
                            "model": f"{m1} vs {m2}",
                            "prompt_type": ptype,
                            "mean_cosine_distance": np.mean(dists),
                            "std_cosine_distance": np.std(dists),
                            "n_pairs": len(dists),
                        })

    print(f"  Computed {len(records)} distance comparisons.")
    return records


def run_activation_analysis(results_dir: str):
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    act_df = load_all_activations(results_dir)
    if act_df.empty:
        print("No activation data found.")
        return

    print(f"Loaded {len(act_df)} activation records.")

    dim_groups = extract_mid_layer_activations(act_df)
    if not dim_groups:
        print("No valid activation vectors extracted.")
        return

    total_vecs = sum(len(m) for _, (_, m) in dim_groups.items())
    print(f"Extracted {total_vecs} activation vectors across {len(dim_groups)} hidden dims: "
          f"{sorted(dim_groups.keys())}")

    # For PCA/UMAP: analyze within each dim group (models sharing same hidden_dim)
    # This is the most meaningful comparison since same-dim often means same-family pairs
    all_dist_records = []
    for dim, (vectors, metadata) in sorted(dim_groups.items()):
        models_in_group = sorted(set(m["model_short_name"] for m in metadata))
        print(f"\n--- Hidden dim {dim}: {len(vectors)} vectors, models: {models_in_group} ---")

        suffix = f"_dim{dim}"
        plot_pca_by_type(vectors, metadata, figures_dir, suffix=suffix)
        plot_umap_by_type(vectors, metadata, figures_dir, suffix=suffix)
        plot_pca_by_model(vectors, metadata, figures_dir, suffix=suffix)
        records = compute_activation_distances(vectors, metadata, results_dir, suffix=suffix)
        if records:
            all_dist_records.extend(records)

    # Save combined distance CSV
    if all_dist_records:
        dist_df = pd.DataFrame(all_dist_records)
        path = Path(results_dir) / "activation_distances.csv"
        dist_df.to_csv(path, index=False)
        print(f"\nSaved combined: {path}")

    # Also create a single combined PCA using the largest dim group for the main figures
    largest_dim = max(dim_groups.keys(), key=lambda d: len(dim_groups[d][1]))
    vectors_main, metadata_main = dim_groups[largest_dim]
    print(f"\nMain figures use dim={largest_dim} ({len(vectors_main)} vectors)")
    plot_pca_by_type(vectors_main, metadata_main, figures_dir, suffix="")
    plot_pca_by_model(vectors_main, metadata_main, figures_dir, suffix="")


def main():
    parser = argparse.ArgumentParser(description="Activation geometry analysis")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()
    run_activation_analysis(args.results_dir)


if __name__ == "__main__":
    main()
