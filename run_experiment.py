#!/usr/bin/env python3
"""Main experiment runner for free generation comparison (base vs RLHF)."""

import argparse
import json
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run free generation experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--prompts", type=str, default="prompts.yaml", help="Path to prompts file")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Which models to run: 'all', a short_name, or a family name",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-completed runs")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: 2 prompts/type, 2 reps, 100 max tokens",
    )
    return parser.parse_args()


def load_config(config_path, prompts_path, quick=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(prompts_path) as f:
        prompts_data = yaml.safe_load(f)

    prompts = prompts_data["prompts"]

    if quick:
        type_counts = {}
        filtered = []
        for p in prompts:
            t = p["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
            if type_counts[t] <= 2:
                filtered.append(p)
        prompts = filtered
        config["generation"]["num_repetitions"] = 2
        config["generation"]["max_new_tokens"] = 100

    return config, prompts


def get_output_paths(output_dir, model_short_name, prompt_id, rep):
    gen_path = Path(output_dir) / "generations" / model_short_name / f"{prompt_id}_rep{rep}.json"
    entropy_path = Path(output_dir) / "entropy" / model_short_name / f"{prompt_id}_rep{rep}.npz"
    act_path = Path(output_dir) / "activations" / model_short_name / f"{prompt_id}_rep{rep}.npz"
    return gen_path, entropy_path, act_path


def is_run_complete(gen_path):
    if not gen_path.exists():
        return False
    try:
        with open(gen_path) as f:
            data = json.load(f)
        return "error" not in data
    except (json.JSONDecodeError, KeyError):
        return False


def get_model_layers(model):
    """Get transformer layers from model, handling different architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Cannot find layers in model architecture: {type(model)}")


def load_model(model_config):
    name = model_config["name"]
    dtype = getattr(torch, model_config["dtype"])
    print(f"Loading model: {name} (dtype={model_config['dtype']})")

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded: {name} ({model.config.num_hidden_layers} layers, "
          f"hidden_dim={model.config.hidden_size})")
    return model, tokenizer


def prepare_input(tokenizer, prompt, is_base, device):
    if is_base:
        input_ids = tokenizer.encode(prompt["text_base"], return_tensors="pt").to(device)
    else:
        messages = [{"role": "user", "content": prompt["text_instruct"]}]
        result = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        input_ids = result["input_ids"].to(device)
    return input_ids


def compute_entropy_profile(scores):
    """Compute token-level entropy and top-k concentration from generation scores."""
    entropies = []
    top_k_concentrations = []
    for score in scores:
        probs = torch.softmax(score, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        entropies.append(entropy.item())
        top_k_probs = torch.topk(probs, k=10, dim=-1).values
        top_k_concentrations.append(top_k_probs.sum(dim=-1).item())
    return np.array(entropies), np.array(top_k_concentrations)


def capture_activations(model, full_input_ids, prompt_length, response_length, config):
    """Capture activations at target layers and positions via a forward pass with hooks."""
    act_config = config["activation_capture"]
    if not act_config["enabled"]:
        return None

    num_layers = model.config.num_hidden_layers
    target_layers = [int(f * (num_layers - 1)) for f in act_config["layer_fractions"]]

    target_positions = []
    for pos in act_config["target_positions"]:
        if pos == -1:
            actual_pos = response_length - 1
        else:
            actual_pos = pos
        if actual_pos < response_length:
            target_positions.append(actual_pos)
    if not target_positions:
        return None

    abs_positions = [prompt_length + p for p in target_positions]

    captured = {}
    layers = get_model_layers(model)

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach()
        return hook_fn

    hooks = []
    for layer_idx in target_layers:
        h = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    try:
        with torch.no_grad():
            model(input_ids=full_input_ids)

        activations = []
        for layer_idx in target_layers:
            if layer_idx not in captured:
                continue
            hidden = captured[layer_idx]
            layer_acts = []
            for abs_pos in abs_positions:
                if abs_pos < hidden.shape[1]:
                    layer_acts.append(hidden[0, abs_pos, :].cpu().float().numpy())
            activations.append(layer_acts)

        if not activations or not activations[0]:
            return None

        activations = np.array(activations)
    finally:
        for h in hooks:
            h.remove()
        del captured
        torch.cuda.empty_cache()

    return {
        "activations": activations,
        "layer_indices": np.array(target_layers),
        "token_positions": np.array(target_positions),
        "response_length": response_length,
    }


def run_single_generation(model, tokenizer, prompt, model_config, config, seed, device):
    """Run a single generation and return results."""
    gen_config = config["generation"]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    is_base = model_config["is_base"]
    input_ids = prepare_input(tokenizer, prompt, is_base, device)
    prompt_length = input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_p=gen_config["top_p"],
            do_sample=gen_config["do_sample"],
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output.sequences[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    num_tokens = len(generated_ids)

    if num_tokens < 20:
        warnings.warn(
            f"Short generation ({num_tokens} tokens) for prompt {prompt['id']} "
            f"with model {model_config['short_name']}"
        )

    entropy_profile, top_k_profile = compute_entropy_profile(output.scores)

    activation_data = None
    try:
        full_sequence = output.sequences
        activation_data = capture_activations(
            model, full_sequence, prompt_length, num_tokens, config
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            warnings.warn(f"OOM during activation capture for {prompt['id']}, skipping activations")
            torch.cuda.empty_cache()
        else:
            raise

    del output
    torch.cuda.empty_cache()

    prompt_text = prompt["text_base"] if is_base else prompt["text_instruct"]

    gen_result = {
        "model": model_config["name"],
        "model_short_name": model_config["short_name"],
        "is_base": is_base,
        "prompt_id": prompt["id"],
        "prompt_type": prompt["type"],
        "prompt_text": prompt_text,
        "repetition": seed - config["random_seed_base"],
        "seed": seed,
        "generated_text": generated_text,
        "num_tokens_generated": num_tokens,
        "generation_config": {
            "temperature": gen_config["temperature"],
            "top_p": gen_config["top_p"],
            "max_new_tokens": gen_config["max_new_tokens"],
        },
        "family": model_config.get("family", "unknown"),
        "size_b": model_config.get("size_b", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    entropy_data = {
        "entropy_profile": entropy_profile,
        "top_k_profile": top_k_profile,
        "token_ids": generated_ids.cpu().numpy(),
    }

    return gen_result, entropy_data, activation_data


def save_results(gen_result, entropy_data, activation_data, gen_path, entropy_path, act_path):
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    entropy_path.parent.mkdir(parents=True, exist_ok=True)

    with open(gen_path, "w") as f:
        json.dump(gen_result, f, indent=2)

    np.savez_compressed(
        entropy_path,
        entropy_profile=entropy_data["entropy_profile"],
        top_k_profile=entropy_data["top_k_profile"],
        token_ids=entropy_data["token_ids"],
    )

    if activation_data is not None:
        act_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            act_path,
            activations=activation_data["activations"],
            layer_indices=activation_data["layer_indices"],
            token_positions=activation_data["token_positions"],
            response_length=activation_data["response_length"],
        )


def save_progress(output_dir, completed):
    progress_path = Path(output_dir) / "progress.json"
    with open(progress_path, "w") as f:
        json.dump({"completed": completed, "last_updated": datetime.now(timezone.utc).isoformat()}, f)


def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    prompts_path = script_dir / args.prompts

    config, prompts = load_config(config_path, prompts_path, quick=args.quick)
    output_dir = script_dir / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter models — support short_name or family name
    models_to_run = config["models"]
    if args.models != "all":
        by_short = [m for m in models_to_run if m["short_name"] == args.models]
        by_family = [m for m in models_to_run if m.get("family") == args.models]
        models_to_run = by_short if by_short else by_family
        if not models_to_run:
            print(f"Error: no model found matching '{args.models}'")
            print(f"Available short_names: {[m['short_name'] for m in config['models']]}")
            print(f"Available families: {sorted(set(m.get('family','') for m in config['models']))}")
            return

    num_reps = config["generation"]["num_repetitions"]
    total_runs = len(models_to_run) * len(prompts) * num_reps

    print(f"Experiment configuration:")
    print(f"  Models: {[m['short_name'] for m in models_to_run]}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Repetitions: {num_reps}")
    print(f"  Total runs: {total_runs}")
    print(f"  Max tokens: {config['generation']['max_new_tokens']}")
    if args.quick:
        print(f"  [QUICK MODE]")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    completed_runs = []
    skipped = 0
    errors = 0
    start_time = time.time()
    runs_done = 0

    for model_config in models_to_run:
        try:
            model, tokenizer = load_model(model_config)
        except Exception as e:
            print(f"ERROR: Failed to load model {model_config['name']}: {e}")
            traceback.print_exc()
            errors += len(prompts) * num_reps
            runs_done += len(prompts) * num_reps
            continue

        for prompt in prompts:
            for rep in range(num_reps):
                seed = config["random_seed_base"] + rep
                gen_path, entropy_path, act_path = get_output_paths(
                    output_dir, model_config["short_name"], prompt["id"], rep
                )

                if args.resume and is_run_complete(gen_path):
                    skipped += 1
                    runs_done += 1
                    continue

                try:
                    gen_result, entropy_data, activation_data = run_single_generation(
                        model, tokenizer, prompt, model_config, config, seed, device
                    )
                    save_results(
                        gen_result, entropy_data, activation_data, gen_path, entropy_path, act_path
                    )
                    completed_runs.append(
                        f"{model_config['short_name']}_{prompt['id']}_rep{rep}"
                    )
                    runs_done += 1

                    elapsed = time.time() - start_time
                    rate = (runs_done - skipped) / elapsed if elapsed > 0 else 0
                    remaining_runs = total_runs - runs_done
                    remaining = remaining_runs / rate if rate > 0 else 0

                    print(
                        f"[{model_config['short_name']}] prompt {prompt['id']} "
                        f"rep {rep + 1}/{num_reps} — "
                        f"generated {gen_result['num_tokens_generated']} tokens "
                        f"({runs_done}/{total_runs}, ~{remaining:.0f}s remaining)"
                    )

                except Exception as e:
                    print(f"ERROR: {model_config['short_name']} {prompt['id']} rep {rep}: {e}")
                    traceback.print_exc()
                    errors += 1
                    runs_done += 1
                    gen_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(gen_path, "w") as f:
                        json.dump(
                            {
                                "model": model_config["name"],
                                "model_short_name": model_config["short_name"],
                                "prompt_id": prompt["id"],
                                "repetition": rep,
                                "seed": seed,
                                "error": str(e),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                            f,
                            indent=2,
                        )

        del model, tokenizer
        torch.cuda.empty_cache()
        print(f"Unloaded {model_config['short_name']}")
        print()

    save_progress(output_dir, completed_runs)
    elapsed = time.time() - start_time
    print(f"\nDone. Completed: {len(completed_runs)}, Skipped: {skipped}, Errors: {errors}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
