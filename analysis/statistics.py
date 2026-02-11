#!/usr/bin/env python3
"""Strengthened statistical analysis: bootstrap CIs, effect sizes, mixed-effects models,
multiple comparison correction, and systematic disagreement analysis."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.formula.api as smf

try:
    from .utils import load_all_generations, load_all_entropy_profiles
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import load_all_generations, load_all_entropy_profiles

import re as re_mod


# ---------------------------------------------------------------------------
# Generic statistical utilities
# ---------------------------------------------------------------------------

def bootstrap_ci(data, statistic_fn=np.mean, n_boot=10000, ci=95, seed=42):
    """Compute bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    boot_stats = np.array([
        statistic_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (100 - ci) / 2
    lo = np.percentile(boot_stats, alpha)
    hi = np.percentile(boot_stats, 100 - alpha)
    return statistic_fn(data), lo, hi


def bootstrap_ci_diff(data_a, data_b, statistic_fn=np.mean, n_boot=10000, ci=95, seed=42):
    """Bootstrap CI for the difference statistic_fn(a) - statistic_fn(b)."""
    rng = np.random.RandomState(seed)
    a = np.asarray(data_a)
    b = np.asarray(data_b)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    diffs = []
    for _ in range(n_boot):
        sa = statistic_fn(rng.choice(a, size=len(a), replace=True))
        sb = statistic_fn(rng.choice(b, size=len(b), replace=True))
        diffs.append(sa - sb)
    diffs = np.array(diffs)
    alpha = (100 - ci) / 2
    lo = np.percentile(diffs, alpha)
    hi = np.percentile(diffs, 100 - alpha)
    return statistic_fn(a) - statistic_fn(b), lo, hi


def cohens_d(group1, group2):
    """Compute Cohen's d effect size (positive = group1 > group2)."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction. Returns corrected p-values."""
    p = np.asarray(p_values)
    n = len(p)
    order = np.argsort(p)
    corrected = np.empty(n)
    for rank, idx in enumerate(order):
        corrected[idx] = min(p[idx] * (n - rank), 1.0)
    # Enforce monotonicity
    for i in range(1, n):
        idx = order[i]
        prev_idx = order[i - 1]
        corrected[idx] = max(corrected[idx], corrected[prev_idx])
    return corrected


def parse_model_info(name):
    """Extract size, family, and instruct status from model short name."""
    size_match = re_mod.search(r'(\d+_?\d*)b', name)
    size = float(size_match.group(1).replace('_', '.')) if size_match else None
    fam = re_mod.sub(r'\d+_?\d*b?_?(base|instruct|it)$', '', name).rstrip('_')
    if not fam:
        fam = name.split("_")[0]
    is_instruct = "instruct" in name or name.endswith("_it")
    return size, fam, is_instruct


# ---------------------------------------------------------------------------
# 1. Content coding statistics
# ---------------------------------------------------------------------------

def compute_content_statistics(results_dir):
    """Bootstrap CIs and effect sizes for content coding scores."""
    results_dir = Path(results_dir)
    coding_dir = results_dir / "coding"

    # Load all coding results
    coding_records = []
    for f in coding_dir.glob("*_coding.json"):
        if "llm_judge" not in f.name:
            with open(f) as fh:
                coding_records.extend(json.load(fh))

    if not coding_records:
        print("No coding data found.")
        return pd.DataFrame()

    df = pd.DataFrame(coding_records)

    # Add model metadata
    df["size_b"] = df["model_short_name"].apply(lambda n: parse_model_info(n)[0])
    df["family"] = df["model_short_name"].apply(lambda n: parse_model_info(n)[1])
    df["is_instruct"] = df["model_short_name"].apply(lambda n: parse_model_info(n)[2])

    records = []

    for ptype in sorted(df["prompt_type"].unique()):
        sub = df[df["prompt_type"] == ptype]
        base = sub[~sub["is_instruct"]]["self_reference"].values
        inst = sub[sub["is_instruct"]]["self_reference"].values

        if len(base) > 0 and len(inst) > 0:
            # Bootstrap CI for base mean
            base_mean, base_lo, base_hi = bootstrap_ci(base)
            inst_mean, inst_lo, inst_hi = bootstrap_ci(inst)

            # Bootstrap CI for difference
            diff, diff_lo, diff_hi = bootstrap_ci_diff(inst, base)

            # Effect size
            d = cohens_d(inst, base)

            # Mann-Whitney U
            u_stat, p_val = sp_stats.mannwhitneyu(inst, base, alternative="two-sided")

            records.append({
                "prompt_type": ptype,
                "metric": "self_reference",
                "base_mean": base_mean,
                "base_ci_lo": base_lo,
                "base_ci_hi": base_hi,
                "instruct_mean": inst_mean,
                "instruct_ci_lo": inst_lo,
                "instruct_ci_hi": inst_hi,
                "diff_mean": diff,
                "diff_ci_lo": diff_lo,
                "diff_ci_hi": diff_hi,
                "cohens_d": d,
                "mann_whitney_U": u_stat,
                "p_value": p_val,
                "n_base": len(base),
                "n_instruct": len(inst),
            })

        # Same for meta_cognition and hedging
        for metric in ["meta_cognition", "hedging"]:
            base_vals = sub[~sub["is_instruct"]][metric].values
            inst_vals = sub[sub["is_instruct"]][metric].values
            if len(base_vals) > 0 and len(inst_vals) > 0:
                base_mean, base_lo, base_hi = bootstrap_ci(base_vals)
                inst_mean, inst_lo, inst_hi = bootstrap_ci(inst_vals)
                diff, diff_lo, diff_hi = bootstrap_ci_diff(inst_vals, base_vals)
                d = cohens_d(inst_vals, base_vals)
                u_stat, p_val = sp_stats.mannwhitneyu(inst_vals, base_vals, alternative="two-sided")
                records.append({
                    "prompt_type": ptype,
                    "metric": metric,
                    "base_mean": base_mean,
                    "base_ci_lo": base_lo,
                    "base_ci_hi": base_hi,
                    "instruct_mean": inst_mean,
                    "instruct_ci_lo": inst_lo,
                    "instruct_ci_hi": inst_hi,
                    "diff_mean": diff,
                    "diff_ci_lo": diff_lo,
                    "diff_ci_hi": diff_hi,
                    "cohens_d": d,
                    "mann_whitney_U": u_stat,
                    "p_value": p_val,
                    "n_base": len(base_vals),
                    "n_instruct": len(inst_vals),
                })

    stats_df = pd.DataFrame(records)

    # Apply Holm-Bonferroni correction
    if not stats_df.empty:
        stats_df["p_corrected"] = holm_bonferroni(stats_df["p_value"].values)

    return stats_df


# ---------------------------------------------------------------------------
# 2. Mixed-effects model
# ---------------------------------------------------------------------------

def fit_mixed_effects(results_dir):
    """Fit a mixed-effects model: self_reference ~ is_instruct * prompt_type + (1|family)."""
    results_dir = Path(results_dir)
    coding_dir = results_dir / "coding"

    coding_records = []
    for f in coding_dir.glob("*_coding.json"):
        if "llm_judge" not in f.name:
            with open(f) as fh:
                coding_records.extend(json.load(fh))

    if not coding_records:
        return None

    df = pd.DataFrame(coding_records)
    df["size_b"] = df["model_short_name"].apply(lambda n: parse_model_info(n)[0])
    df["family"] = df["model_short_name"].apply(lambda n: parse_model_info(n)[1])
    df["is_instruct"] = df["model_short_name"].apply(lambda n: parse_model_info(n)[2]).astype(int)
    df["log_size"] = np.log2(df["size_b"])

    results = {}

    # Model 1: self_reference ~ is_instruct + (1|family)
    try:
        md1 = smf.mixedlm("self_reference ~ is_instruct", df, groups=df["family"])
        mdf1 = md1.fit(reml=True)
        results["model1_self_ref_instruct"] = {
            "formula": "self_reference ~ is_instruct + (1|family)",
            "instruct_coef": float(mdf1.params.get("is_instruct", np.nan)),
            "instruct_pval": float(mdf1.pvalues.get("is_instruct", np.nan)),
            "instruct_ci_lo": float(mdf1.conf_int().loc["is_instruct", 0]) if "is_instruct" in mdf1.conf_int().index else np.nan,
            "instruct_ci_hi": float(mdf1.conf_int().loc["is_instruct", 1]) if "is_instruct" in mdf1.conf_int().index else np.nan,
            "group_var": float(mdf1.cov_re.iloc[0, 0]) if hasattr(mdf1, 'cov_re') else np.nan,
            "aic": float(mdf1.aic) if hasattr(mdf1, 'aic') else np.nan,
            "n": len(df),
            "converged": mdf1.converged,
        }
    except Exception as e:
        results["model1_self_ref_instruct"] = {"error": str(e)}

    # Model 2: self_reference ~ is_instruct * prompt_type + (1|family)
    try:
        # Only self_reference and structured_novelty prompts (skip topic_control as baseline is ~0)
        df_sr = df[df["prompt_type"].isin(["self_reference", "structured_novelty", "unconstrained"])]
        md2 = smf.mixedlm(
            "self_reference ~ is_instruct * C(prompt_type, Treatment(reference='unconstrained'))",
            df_sr, groups=df_sr["family"]
        )
        mdf2 = md2.fit(reml=True)
        results["model2_interaction"] = {
            "formula": "self_reference ~ is_instruct * prompt_type + (1|family)",
            "params": {k: float(v) for k, v in mdf2.params.items()},
            "pvalues": {k: float(v) for k, v in mdf2.pvalues.items()},
            "n": len(df_sr),
            "converged": mdf2.converged,
        }
    except Exception as e:
        results["model2_interaction"] = {"error": str(e)}

    # Model 3: self_reference ~ is_instruct + log_size + (1|family)
    try:
        md3 = smf.mixedlm("self_reference ~ is_instruct + log_size", df, groups=df["family"])
        mdf3 = md3.fit(reml=True)
        results["model3_with_size"] = {
            "formula": "self_reference ~ is_instruct + log_size + (1|family)",
            "params": {k: float(v) for k, v in mdf3.params.items()},
            "pvalues": {k: float(v) for k, v in mdf3.pvalues.items()},
            "n": len(df),
            "converged": mdf3.converged,
        }
    except Exception as e:
        results["model3_with_size"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# 3. Enhanced entropy statistics
# ---------------------------------------------------------------------------

def compute_entropy_statistics_enhanced(results_dir):
    """Enhanced entropy stats with bootstrap CIs and effect sizes."""
    results_dir = Path(results_dir)
    entropy_df = load_all_entropy_profiles(results_dir)
    if entropy_df.empty:
        return pd.DataFrame()

    entropy_df["is_instruct"] = entropy_df["model_short_name"].apply(
        lambda n: parse_model_info(n)[2]
    )

    records = []
    for ptype in sorted(entropy_df["prompt_type"].unique()):
        sub = entropy_df[entropy_df["prompt_type"] == ptype]

        # Mean entropy per response
        base_means = np.array([np.mean(ep) for ep in sub[~sub["is_instruct"]]["entropy_profile"]])
        inst_means = np.array([np.mean(ep) for ep in sub[sub["is_instruct"]]["entropy_profile"]])

        if len(base_means) > 0 and len(inst_means) > 0:
            bm, blo, bhi = bootstrap_ci(base_means)
            im, ilo, ihi = bootstrap_ci(inst_means)
            diff, dlo, dhi = bootstrap_ci_diff(base_means, inst_means)
            d = cohens_d(base_means, inst_means)
            u, p = sp_stats.mannwhitneyu(base_means, inst_means, alternative="two-sided")

            records.append({
                "prompt_type": ptype,
                "metric": "mean_entropy",
                "base_mean": bm, "base_ci": f"[{blo:.4f}, {bhi:.4f}]",
                "instruct_mean": im, "instruct_ci": f"[{ilo:.4f}, {ihi:.4f}]",
                "diff": diff, "diff_ci": f"[{dlo:.4f}, {dhi:.4f}]",
                "cohens_d": d,
                "U_statistic": u, "p_value": p,
            })

        # Initial entropy (first token)
        base_init = np.array([ep[0] for ep in sub[~sub["is_instruct"]]["entropy_profile"] if len(ep) > 0])
        inst_init = np.array([ep[0] for ep in sub[sub["is_instruct"]]["entropy_profile"] if len(ep) > 0])

        if len(base_init) > 0 and len(inst_init) > 0:
            bm, blo, bhi = bootstrap_ci(base_init)
            im, ilo, ihi = bootstrap_ci(inst_init)
            diff, dlo, dhi = bootstrap_ci_diff(base_init, inst_init)
            d = cohens_d(base_init, inst_init)
            u, p = sp_stats.mannwhitneyu(base_init, inst_init, alternative="two-sided")

            records.append({
                "prompt_type": ptype,
                "metric": "initial_entropy",
                "base_mean": bm, "base_ci": f"[{blo:.4f}, {bhi:.4f}]",
                "instruct_mean": im, "instruct_ci": f"[{ilo:.4f}, {ihi:.4f}]",
                "diff": diff, "diff_ci": f"[{dlo:.4f}, {dhi:.4f}]",
                "cohens_d": d,
                "U_statistic": u, "p_value": p,
            })

    stats_df = pd.DataFrame(records)
    if not stats_df.empty:
        stats_df["p_corrected"] = holm_bonferroni(stats_df["p_value"].values)

    return stats_df


# ---------------------------------------------------------------------------
# 4. Scaling statistics
# ---------------------------------------------------------------------------

def compute_scaling_statistics(results_dir):
    """Test scaling trends with Spearman correlations and bootstrap CIs on slopes."""
    results_dir = Path(results_dir)
    coding_dir = results_dir / "coding"
    summary_path = coding_dir / "summary.csv"

    if not summary_path.exists():
        return {}

    summary = pd.read_csv(summary_path)
    summary["size_b"] = summary["model"].apply(lambda n: parse_model_info(n)[0])
    summary["family"] = summary["model"].apply(lambda n: parse_model_info(n)[1])
    summary["is_instruct"] = summary["model"].apply(lambda n: parse_model_info(n)[2])

    results = {}

    for ptype in ["self_reference", "structured_novelty"]:
        sub = summary[summary["prompt_type"] == ptype]

        for model_type in ["base", "instruct"]:
            type_data = sub[sub["is_instruct"] == (model_type == "instruct")]

            for fam in sorted(type_data["family"].unique()):
                fam_data = type_data[type_data["family"] == fam].sort_values("size_b")
                if len(fam_data) < 2:
                    continue

                sizes = fam_data["size_b"].values
                scores = fam_data["mean_self_reference"].values

                # Spearman correlation
                rho, p_val = sp_stats.spearmanr(sizes, scores)

                # Bootstrap CI on slope (linear regression)
                log_sizes = np.log2(sizes)
                def slope_fn(data):
                    x = log_sizes[np.array(data, dtype=int)]
                    y = scores[np.array(data, dtype=int)]
                    if len(x) < 2:
                        return 0.0
                    slope, _, _, _, _ = sp_stats.linregress(x, y)
                    return slope

                # Simple bootstrap on the slope
                rng = np.random.RandomState(42)
                indices = np.arange(len(sizes))
                boot_slopes = []
                for _ in range(5000):
                    boot_idx = rng.choice(indices, size=len(indices), replace=True)
                    x_boot = log_sizes[boot_idx]
                    y_boot = scores[boot_idx]
                    if len(np.unique(x_boot)) >= 2:
                        s, _, _, _, _ = sp_stats.linregress(x_boot, y_boot)
                        boot_slopes.append(s)
                boot_slopes = np.array(boot_slopes) if boot_slopes else np.array([0.0])

                actual_slope = sp_stats.linregress(log_sizes, scores).slope if len(np.unique(log_sizes)) >= 2 else 0.0

                key = f"{ptype}_{fam}_{model_type}"
                results[key] = {
                    "prompt_type": ptype,
                    "family": fam,
                    "model_type": model_type,
                    "n_sizes": len(sizes),
                    "sizes": sizes.tolist(),
                    "scores": scores.tolist(),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_val),
                    "slope_per_log2B": float(actual_slope),
                    "slope_ci_lo": float(np.percentile(boot_slopes, 2.5)),
                    "slope_ci_hi": float(np.percentile(boot_slopes, 97.5)),
                }

    return results


# ---------------------------------------------------------------------------
# 5. Agreement / disagreement analysis
# ---------------------------------------------------------------------------

def compute_agreement_statistics(results_dir):
    """Analyze systematic disagreements between keyword coder and LLM judge."""
    results_dir = Path(results_dir)
    coding_dir = results_dir / "coding"

    # Load keyword coding
    kw_records = []
    for f in coding_dir.glob("*_coding.json"):
        if "llm_judge" not in f.name:
            with open(f) as fh:
                kw_records.extend(json.load(fh))

    judge_path = coding_dir / "llm_judge_all.json"
    if not judge_path.exists() or not kw_records:
        return {}

    kw_df = pd.DataFrame(kw_records)
    with open(judge_path) as f:
        judge_records = json.load(f)
    judge_df = pd.DataFrame(judge_records)

    # Merge
    merged = kw_df.merge(
        judge_df[["model_short_name", "prompt_id", "repetition",
                   "self_reference", "meta_cognition", "hedging", "introspective_depth"]],
        on=["model_short_name", "prompt_id", "repetition"],
        how="inner",
        suffixes=("_kw", "_judge"),
    )

    if merged.empty:
        return {}

    merged["is_instruct"] = merged["model_short_name"].apply(lambda n: parse_model_info(n)[2])
    merged["family"] = merged["model_short_name"].apply(lambda n: parse_model_info(n)[1])

    results = {}

    # Overall agreement metrics per dimension
    for dim in ["self_reference", "meta_cognition", "hedging"]:
        kw_col = f"{dim}_kw"
        judge_col = f"{dim}_judge"

        if kw_col not in merged.columns or judge_col not in merged.columns:
            continue

        valid = merged[[kw_col, judge_col]].dropna()
        kw_vals = valid[kw_col].values
        judge_vals = valid[judge_col].values

        # Pearson r
        r, r_p = sp_stats.pearsonr(kw_vals, judge_vals) if len(valid) > 2 else (np.nan, np.nan)

        # Mean signed difference (judge - keyword): positive = judge rates higher
        diffs = judge_vals - kw_vals
        bias_mean, bias_lo, bias_hi = bootstrap_ci(diffs)

        # By model type
        base_mask = ~merged.loc[valid.index, "is_instruct"].values
        inst_mask = merged.loc[valid.index, "is_instruct"].values

        base_bias = np.mean(diffs[base_mask]) if base_mask.sum() > 0 else np.nan
        inst_bias = np.mean(diffs[inst_mask]) if inst_mask.sum() > 0 else np.nan

        # Exact agreement rate
        exact = np.mean(kw_vals == judge_vals)

        # Within-1 agreement
        within1 = np.mean(np.abs(diffs) <= 1)

        results[dim] = {
            "pearson_r": float(r),
            "pearson_p": float(r_p),
            "bias_mean": float(bias_mean),
            "bias_ci": f"[{bias_lo:.3f}, {bias_hi:.3f}]",
            "bias_base": float(base_bias),
            "bias_instruct": float(inst_bias),
            "exact_agreement": float(exact),
            "within_1_agreement": float(within1),
            "n": len(valid),
        }

    # Judge-only metric: introspective_depth by condition
    if "introspective_depth" in merged.columns:
        depth_stats = []
        for ptype in sorted(merged["prompt_type"].unique()):
            for is_inst in [False, True]:
                sub = merged[(merged["prompt_type"] == ptype) & (merged["is_instruct"] == is_inst)]
                if len(sub) > 0 and "introspective_depth" in sub.columns:
                    vals = sub["introspective_depth"].dropna().values
                    if len(vals) > 0:
                        m, lo, hi = bootstrap_ci(vals)
                        depth_stats.append({
                            "prompt_type": ptype,
                            "type": "instruct" if is_inst else "base",
                            "mean": float(m),
                            "ci_lo": float(lo),
                            "ci_hi": float(hi),
                            "n": len(vals),
                        })
        results["introspective_depth_by_condition"] = depth_stats

    return results


# ---------------------------------------------------------------------------
# 6. Probing significance tests
# ---------------------------------------------------------------------------

def compute_probing_statistics(results_dir):
    """Add permutation-based significance to probing results."""
    results_dir = Path(results_dir)
    probe_path = results_dir / "probing_results.json"
    if not probe_path.exists():
        return []

    with open(probe_path) as f:
        probe_data = json.load(f)

    # For each probe, compute if accuracy significantly exceeds baseline
    # using a binomial test approximation
    enhanced = []
    for r in probe_data:
        if "error" in r:
            enhanced.append(r)
            continue

        acc = r.get("accuracy_mean", 0)
        baseline = r.get("majority_baseline", 0.5)
        n = r.get("n_samples", 0)
        n_folds = 5  # from StratifiedKFold

        # Approximate: total test predictions = n (each sample tested once in CV)
        # Under null (chance = baseline), n_correct ~ Binomial(n, baseline)
        n_correct = int(acc * n)
        # One-sided binomial test: is accuracy significantly above baseline?
        p_val = sp_stats.binomtest(n_correct, n, baseline, alternative="greater").pvalue if n > 0 else 1.0

        # Effect size: lift over baseline normalized by baseline std
        # Std of binomial proportion = sqrt(p(1-p)/n)
        baseline_std = np.sqrt(baseline * (1 - baseline) / n) if n > 0 else 1.0
        z_score = (acc - baseline) / baseline_std if baseline_std > 0 else 0.0

        r_enhanced = dict(r)
        r_enhanced["p_value_vs_baseline"] = float(p_val)
        r_enhanced["z_score"] = float(z_score)
        enhanced.append(r_enhanced)

    # Holm-Bonferroni correction across all valid probes
    valid_probes = [r for r in enhanced if "error" not in r and "p_value_vs_baseline" in r]
    if valid_probes:
        p_vals = [r["p_value_vs_baseline"] for r in valid_probes]
        corrected = holm_bonferroni(p_vals)
        for r, pc in zip(valid_probes, corrected):
            r["p_corrected"] = float(pc)

    return enhanced


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def run_all_statistics(results_dir):
    """Run all strengthened statistical analyses and save results."""
    results_dir = Path(results_dir)
    stats_dir = results_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STRENGTHENED STATISTICAL ANALYSIS")
    print("=" * 60)

    # 1. Content statistics with bootstrap CIs
    print("\n--- 1. Content Coding Statistics ---")
    content_stats = compute_content_statistics(results_dir)
    if not content_stats.empty:
        path = stats_dir / "content_statistics.csv"
        content_stats.to_csv(path, index=False)
        print(f"Saved: {path}")
        print(f"  {len(content_stats)} comparisons, "
              f"{(content_stats['p_corrected'] < 0.05).sum()} significant after Holm-Bonferroni")

        # Print key results
        sr = content_stats[
            (content_stats["metric"] == "self_reference")
            & (content_stats["prompt_type"] == "self_reference")
        ]
        if not sr.empty:
            row = sr.iloc[0]
            print(f"  Self-ref on self_reference prompts: "
                  f"instruct={row['instruct_mean']:.3f} [{row['instruct_ci_lo']:.3f}, {row['instruct_ci_hi']:.3f}] "
                  f"vs base={row['base_mean']:.3f} [{row['base_ci_lo']:.3f}, {row['base_ci_hi']:.3f}]")
            print(f"  Cohen's d={row['cohens_d']:.3f}, p={row['p_value']:.2e}, p_corrected={row['p_corrected']:.2e}")

    # 2. Mixed-effects models
    print("\n--- 2. Mixed-Effects Models ---")
    me_results = fit_mixed_effects(results_dir)
    if me_results:
        path = stats_dir / "mixed_effects.json"
        with open(path, "w") as f:
            json.dump(me_results, f, indent=2, default=str)
        print(f"Saved: {path}")
        for name, res in me_results.items():
            if "error" in res:
                print(f"  {name}: ERROR - {res['error']}")
            else:
                print(f"  {name}: converged={res.get('converged', 'N/A')}")
                if "instruct_coef" in res:
                    print(f"    instruct coefficient={res['instruct_coef']:.4f}, "
                          f"p={res['instruct_pval']:.2e}")

    # 3. Enhanced entropy statistics
    print("\n--- 3. Enhanced Entropy Statistics ---")
    entropy_stats = compute_entropy_statistics_enhanced(results_dir)
    if not entropy_stats.empty:
        path = stats_dir / "entropy_statistics_enhanced.csv"
        entropy_stats.to_csv(path, index=False)
        print(f"Saved: {path}")
        for _, row in entropy_stats.iterrows():
            print(f"  {row['prompt_type']} {row['metric']}: "
                  f"d={row['cohens_d']:.2f}, p={row['p_value']:.2e}")

    # 4. Scaling statistics
    print("\n--- 4. Scaling Statistics ---")
    scaling_stats = compute_scaling_statistics(results_dir)
    if scaling_stats:
        path = stats_dir / "scaling_statistics.json"
        with open(path, "w") as f:
            json.dump(scaling_stats, f, indent=2, default=str)
        print(f"Saved: {path}")
        for key, res in scaling_stats.items():
            if res["n_sizes"] >= 3:
                print(f"  {key}: rho={res['spearman_rho']:.3f} (p={res['spearman_p']:.3f}), "
                      f"slope={res['slope_per_log2B']:.4f} [{res['slope_ci_lo']:.4f}, {res['slope_ci_hi']:.4f}]")

    # 5. Agreement statistics
    print("\n--- 5. Agreement Statistics ---")
    agreement_stats = compute_agreement_statistics(results_dir)
    if agreement_stats:
        path = stats_dir / "agreement_statistics.json"
        with open(path, "w") as f:
            json.dump(agreement_stats, f, indent=2, default=str)
        print(f"Saved: {path}")
        for dim, metrics in agreement_stats.items():
            if isinstance(metrics, dict) and "pearson_r" in metrics:
                print(f"  {dim}: r={metrics['pearson_r']:.3f}, bias={metrics['bias_mean']:.3f} "
                      f"{metrics['bias_ci']}, exact={metrics['exact_agreement']:.1%}")

    # 6. Probing significance
    print("\n--- 6. Probing Significance ---")
    probe_stats = compute_probing_statistics(results_dir)
    if probe_stats:
        path = stats_dir / "probing_statistics.json"
        with open(path, "w") as f:
            json.dump(probe_stats, f, indent=2, default=str)
        print(f"Saved: {path}")
        valid = [r for r in probe_stats if "error" not in r]
        sig = [r for r in valid if r.get("p_corrected", 1) < 0.05]
        print(f"  {len(valid)} probes, {len(sig)} significant after correction")

    print(f"\nAll statistics saved to {stats_dir}/")
    return {
        "content": content_stats,
        "mixed_effects": me_results,
        "entropy": entropy_stats,
        "scaling": scaling_stats,
        "agreement": agreement_stats,
        "probing": probe_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Strengthened statistical analysis")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()
    run_all_statistics(args.results_dir)


if __name__ == "__main__":
    main()
