"""Shared measurement utilities for analysis scripts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_generation(path: str) -> dict:
    """Load a single generation result JSON."""
    with open(path) as f:
        return json.load(f)


def load_all_generations(results_dir: str) -> pd.DataFrame:
    """Load all generation results into a DataFrame."""
    gen_dir = Path(results_dir) / "generations"
    records = []
    for model_dir in sorted(gen_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for json_file in sorted(model_dir.glob("*.json")):
            data = load_generation(json_file)
            if "error" in data:
                continue
            records.append({
                "model": data["model"],
                "model_short_name": data["model_short_name"],
                "is_base": data["is_base"],
                "prompt_id": data["prompt_id"],
                "prompt_type": data["prompt_type"],
                "prompt_text": data.get("prompt_text", ""),
                "repetition": data["repetition"],
                "generated_text": data["generated_text"],
                "num_tokens_generated": data["num_tokens_generated"],
                "seed": data["seed"],
                "family": data.get("family", "unknown"),
                "size_b": data.get("size_b", 0),
            })
    return pd.DataFrame(records)


def load_entropy_profile(path: str) -> dict:
    """Load entropy and top-k profiles from npz."""
    data = np.load(path)
    return {
        "entropy_profile": data["entropy_profile"],
        "top_k_profile": data["top_k_profile"],
        "token_ids": data["token_ids"],
    }


def load_all_entropy_profiles(results_dir: str) -> pd.DataFrame:
    """Load all entropy profiles into a DataFrame."""
    entropy_dir = Path(results_dir) / "entropy"
    records = []
    for model_dir in sorted(entropy_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_short_name = model_dir.name
        for npz_file in sorted(model_dir.glob("*.npz")):
            # Parse prompt_id and rep from filename: {prompt_id}_rep{N}.npz
            stem = npz_file.stem
            parts = stem.rsplit("_rep", 1)
            if len(parts) != 2:
                continue
            prompt_id = parts[0]
            rep = int(parts[1])

            data = load_entropy_profile(npz_file)

            # Look up prompt_type from generations
            gen_path = Path(results_dir) / "generations" / model_short_name / f"{stem}.json"
            prompt_type = "unknown"
            if gen_path.exists():
                gen_data = load_generation(gen_path)
                prompt_type = gen_data.get("prompt_type", "unknown")

            records.append({
                "model_short_name": model_short_name,
                "prompt_id": prompt_id,
                "prompt_type": prompt_type,
                "repetition": rep,
                "entropy_profile": data["entropy_profile"],
                "top_k_profile": data["top_k_profile"],
            })
    return pd.DataFrame(records)


def load_activations(path: str) -> dict:
    """Load activation snapshots from npz."""
    data = np.load(path)
    return {
        "activations": data["activations"],
        "layer_indices": data["layer_indices"],
        "token_positions": data["token_positions"],
        "response_length": int(data["response_length"]),
    }


def load_all_activations(results_dir: str) -> pd.DataFrame:
    """Load all activation data into a DataFrame."""
    act_dir = Path(results_dir) / "activations"
    if not act_dir.exists():
        return pd.DataFrame()
    records = []
    for model_dir in sorted(act_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_short_name = model_dir.name
        for npz_file in sorted(model_dir.glob("*.npz")):
            stem = npz_file.stem
            parts = stem.rsplit("_rep", 1)
            if len(parts) != 2:
                continue
            prompt_id = parts[0]
            rep = int(parts[1])

            data = load_activations(npz_file)

            # Look up prompt_type
            gen_path = Path(results_dir) / "generations" / model_short_name / f"{stem}.json"
            prompt_type = "unknown"
            if gen_path.exists():
                gen_data = load_generation(gen_path)
                prompt_type = gen_data.get("prompt_type", "unknown")

            records.append({
                "model_short_name": model_short_name,
                "prompt_id": prompt_id,
                "prompt_type": prompt_type,
                "repetition": rep,
                "activations": data["activations"],
                "layer_indices": data["layer_indices"],
                "token_positions": data["token_positions"],
                "response_length": data["response_length"],
            })
    return pd.DataFrame(records)


def compute_response_similarity(text_a: str, text_b: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Compute semantic similarity between two responses using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode([text_a, text_b])
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(similarity)


def compute_pairwise_similarity(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Compute pairwise similarity matrix for a list of texts."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings @ embeddings.T
