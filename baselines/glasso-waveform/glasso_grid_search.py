import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import graphical_lasso

from synthetic_data_generator import DataGenerator


# =============================================================================
# Mathematical Helpers
# =============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def get_path_adjacency(N: int) -> np.ndarray:
    """Creates the exact adjacency matrix for a 1D aperiodic path graph."""
    A = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        if i > 0:
            A[i, i - 1] = 1.0
        if i < N - 1:
            A[i, i + 1] = 1.0

    return A


def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def sorted_eigh(M: np.ndarray, descending: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition with deterministic eigenvalue ordering."""
    vals, vecs = np.linalg.eigh(symmetrize(M))
    idx = np.argsort(vals)
    if descending:
        idx = idx[::-1]
    return vals[idx], vecs[:, idx]


def fit_observed_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a coordinatewise standardizer and returns standardized data.

    The previous version computed the covariance on standardized observations but
    then applied the learned map to raw observations. Here we use the same
    standardized representation for fitting and validation, which usually makes
    the baselines more stable.
    """
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    X_std = (X - mean) / std
    return X_std, mean, std


def apply_observed_standardizer(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (X - mean) / std


def compute_robust_covariance_from_standardized(
    X_std: np.ndarray,
    diagonal_loading: float = 1e-2,
) -> np.ndarray:
    """Computes a robust SPD empirical covariance matrix from standardized data."""
    S = np.cov(X_std, rowvar=False)
    S = symmetrize(S)
    np.fill_diagonal(S, S.diagonal() + diagonal_loading)
    return S


def get_feature_config(feature: str):
    feat_gaussian = [
        {
            "type": "gaussian",
            "scale_min": 0.5,
            "scale_max": 2.5,
            "amplitude_min": 0.5,
            "amplitude_max": 2.5,
        }
    ]

    feat_legendre = [
        {
            "type": "legendre",
            "scale_min": 6.0,
            "scale_max": 15.0,
            "l": 3,
            "m": 1,
            "amplitude_min": 0.5,
            "amplitude_max": 2.5,
        },
        {
            "type": "legendre",
            "scale_min": 6.0,
            "scale_max": 15.0,
            "l": 2,
            "m": 1,
            "amplitude_min": 0.5,
            "amplitude_max": 2.5,
        },
    ]

    feat_ising = [
        {
            "type": "ising",
            "beta_min": 1.0,
            "beta_max": 2.5,
            "n_gibbs_steps": 10,
        }
    ]

    if feature == "gaussian":
        return feat_gaussian
    if feature == "legendre":
        return feat_legendre
    if feature == "ising":
        return feat_ising

    raise ValueError(f"Undefined feature type: {feature}")


def get_num_of_lots(feature: str) -> int:
    """
    Feature-specific DataGenerator setting.

    For the waveform experiments, Gaussian/Legendre samples are generated as
    superpositions of several localized lots. The Ising chain is already a full
    chain-valued sample, so we keep the legacy waveform setting num_of_lots=1.
    """
    return 1 if feature == "ising" else 5


def should_use_orthogonal_alignment(output_representation: str) -> bool:
    return output_representation in ["natural", "dst", "permuted"]


# =============================================================================
# Baselines
# =============================================================================

def discover_glasso_topology_and_match_linear(
    S: np.ndarray,
    A_chain: np.ndarray,
    alpha: float,
    orthogonal: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    GLASSO baseline.

    1. Estimate a sparse precision matrix.
    2. Interpret its off-diagonal magnitude as a graph-like operator.
    3. Match its eigenbasis to the path graph eigenbasis.
    """
    _, precision_ = graphical_lasso(
        emp_cov=S,
        alpha=alpha,
        max_iter=300,
    )

    Theta_glasso = np.abs(precision_)
    np.fill_diagonal(Theta_glasso, 0.0)

    # Soft thresholding to reduce numerical noise before eigendecomposition.
    Theta_glasso = np.where(Theta_glasso > 1e-3, Theta_glasso, 0.0)

    vals_g, vecs_g = sorted_eigh(Theta_glasso, descending=True)
    vals_t, vecs_t = sorted_eigh(A_chain, descending=True)

    if orthogonal:
        W = vecs_g @ vecs_t.T
    else:
        scale = np.sqrt(np.abs(vals_t) / (np.abs(vals_g) + 1e-8))
        W = vecs_g @ np.diag(scale) @ vecs_t.T

    frob_error = np.linalg.norm(
        Theta_glasso - W @ A_chain @ W.T,
        ord="fro",
    )

    return W, float(frob_error)


def make_smooth_path_spectrum(
    N: int,
    gamma: float,
    power: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Creates a generic smooth low-pass covariance spectrum on a 1D path.

    The exact hidden covariance is not used. The only assumption is that the
    latent signal is smooth/local on a 1D domain, so low graph frequencies should
    dominate high graph frequencies.
    """
    if N <= 1:
        return np.ones(N, dtype=float)

    normalized_frequency = np.linspace(0.0, 1.0, N)
    spectrum = (1.0 + gamma * normalized_frequency**2) ** (-power)
    spectrum = np.maximum(spectrum, eps)

    # Put the spectrum on roughly the same scale as standardized covariance.
    spectrum = spectrum / (np.mean(spectrum) + eps)
    return spectrum


def discover_spectral_covariance_matching(
    S: np.ndarray,
    A_chain: np.ndarray,
    mode: str = "recolor",
    gamma: float = 1.0,
    power: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Spectral Covariance Matching baseline.

    This is a stronger second-order baseline than GLASSO for smooth stationary
    signals. Instead of estimating sparse conditional dependencies, it directly
    aligns the empirical covariance eigenbasis with the path-graph spectral basis.

    Modes:
        rotate:
            PCA/path-basis rotation only. Strong for orthogonal output
            representations when the empirical covariance spectrum is informative.

        whiten:
            PCA whitening followed by path-basis rotation. This is less faithful
            to signal amplitudes but can help when scale dominates.

        recolor:
            PCA whitening followed by recoloring with a generic smooth path
            covariance spectrum. This is usually the strongest unsupervised
            classical baseline for smooth GSN-like latent signals.
    """
    vals_obs, vecs_obs = sorted_eigh(S, descending=True)
    vals_obs = np.maximum(vals_obs, eps)

    _, vecs_path = sorted_eigh(A_chain, descending=True)

    if mode == "rotate":
        scale = np.ones_like(vals_obs)
    elif mode == "whiten":
        scale = 1.0 / np.sqrt(vals_obs)
    elif mode == "recolor":
        reference_spectrum = make_smooth_path_spectrum(
            N=S.shape[0],
            gamma=gamma,
            power=power,
            eps=eps,
        )
        scale = np.sqrt(reference_spectrum / vals_obs)
    else:
        raise ValueError(f"Unknown spectral mode: {mode}")

    W = vecs_obs @ np.diag(scale) @ vecs_path.T

    info = {
        "mode": mode,
        "gamma": float(gamma),
        "power": float(power),
        "min_empirical_eigenvalue": float(np.min(vals_obs)),
        "max_empirical_eigenvalue": float(np.max(vals_obs)),
        "empirical_condition_number": float(np.max(vals_obs) / np.min(vals_obs)),
    }

    return W, info


# =============================================================================
# Metrics
# =============================================================================

def compute_max_alignment_correlation_gsn(
    sig_orig: np.ndarray,
    sig_rec: np.ndarray,
) -> float:
    """
    Computes max Pearson correlation over shifts, flips, and sign inversions.

    This preserves the natural ambiguities in 1D domain recovery: translation,
    reflection, and global sign.
    """
    std_orig = np.std(sig_orig)
    std_rec = np.std(sig_rec)

    if std_orig < 1e-8 or std_rec < 1e-8:
        return 0.0

    max_corr = 0.0
    N = len(sig_orig)

    for shift in range(N):
        shifted_rec = np.roll(sig_rec, shift)

        corr_normal = np.corrcoef(sig_orig, shifted_rec)[0, 1]
        corr_flipped = np.corrcoef(sig_orig, shifted_rec[::-1])[0, 1]

        if np.isnan(corr_normal):
            corr_normal = 0.0
        if np.isnan(corr_flipped):
            corr_flipped = 0.0

        max_corr = max(max_corr, abs(corr_normal), abs(corr_flipped))

    return float(max_corr)


def summarize_correlations(correlations: np.ndarray) -> Dict[str, float]:
    correlations = np.array(correlations, dtype=float)
    n = len(correlations)

    return {
        "correlation_mean": float(np.mean(correlations)),
        "correlation_std_across_samples": float(np.std(correlations, ddof=1))
        if n > 1
        else 0.0,
        "correlation_variance_across_samples": float(np.var(correlations, ddof=1))
        if n > 1
        else 0.0,
        "correlation_min": float(np.min(correlations)),
        "correlation_max": float(np.max(correlations)),
    }


def evaluate_recovery(
    X_val_orig: np.ndarray,
    X_val_rec: np.ndarray,
) -> Dict[str, float]:
    correlations = [
        compute_max_alignment_correlation_gsn(X_val_orig[i], X_val_rec[i])
        for i in range(len(X_val_orig))
    ]
    return summarize_correlations(np.array(correlations, dtype=float))


# =============================================================================
# Visualization
# =============================================================================

def plot_all_recoveries_gsn(
    X_original: np.ndarray,
    X_observed: np.ndarray,
    recovered_dict: Dict[str, np.ndarray],
    N: int,
    n_plot: int = 5,
    save_path: str = "baseline_gsn_recovery.png",
    observed_label: str = "Observed",
):
    method_labels = list(recovered_dict.keys())
    n_rows = 2 + len(method_labels)

    fig, axes = plt.subplots(
        n_rows,
        n_plot,
        figsize=(15, max(4, 1.8 * n_rows)),
        sharex=True,
        sharey=False,
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    x_axis = np.arange(N)

    for col in range(n_plot):
        ax_orig = axes[0, col]
        ax_orig.plot(x_axis, X_original[col], color="black", linewidth=1.5)
        if col == 0:
            ax_orig.set_ylabel("Original")
        ax_orig.set_title(f"Sample {col + 1}")
        ax_orig.grid(True, linestyle="--", alpha=0.5)

        ax_obs = axes[1, col]
        ax_obs.plot(x_axis, X_observed[col], color="red", linewidth=1.5)
        if col == 0:
            ax_obs.set_ylabel(observed_label)
        ax_obs.grid(True, linestyle="--", alpha=0.5)

        for row_idx, method_label in enumerate(method_labels):
            ax_rec = axes[row_idx + 2, col]
            X_rec = recovered_dict[method_label][col]

            ax_rec.plot(x_axis, X_rec, color="blue", linewidth=1.5)
            if col == 0:
                ax_rec.set_ylabel(method_label[:28])
            ax_rec.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# =============================================================================
# Single Trial
# =============================================================================

def run_single_trial(
    *,
    trial_idx: int,
    seed: int,
    feature: str,
    output_representation: str,
    n_components: int,
    n_total_samples: int,
    batch_size: int,
    n_val_metrics: int,
    baselines: List[str],
    alphas_to_test: List[float],
    spectral_modes: List[str],
    spectral_gammas: List[float],
    spectral_powers: List[float],
    diagonal_loading: float,
    save_plot: bool,
    plot_dir: Path,
    n_plot_methods: int,
):
    set_seed(seed)

    feat = get_feature_config(feature)

    generator = DataGenerator(
        batch_size=batch_size,
        features=feat,
        n_components=n_components,
        num_of_lots=get_num_of_lots(feature),
        is_circulant=False,
        output_representation=output_representation,
    )

    print(
        f"\n[Trial {trial_idx}] "
        f"feature={feature}, "
        f"output_representation={output_representation}, "
        f"seed={seed}"
    )

    print(f"[Trial {trial_idx}] Generating {n_total_samples} samples...")

    X_observed_list = []
    X_original_list = []
    samples_generated = 0

    while samples_generated < n_total_samples:
        X_obs_batch, X_orig_batch = generator.sample_batch_of_data(
            return_hidden_signal=True
        )

        X_observed_list.append(X_obs_batch)
        X_original_list.append(X_orig_batch)
        samples_generated += len(X_obs_batch)

        print(
            f"[Trial {trial_idx}] Generated "
            f"{samples_generated} / {n_total_samples}",
            end="\r",
        )

    print()

    X_observed = np.vstack(X_observed_list)[:n_total_samples]
    X_original = np.vstack(X_original_list)[:n_total_samples]

    print(f"[Trial {trial_idx}] Standardizing observations and computing covariance...")

    X_observed_std, obs_mean, obs_std = fit_observed_standardizer(X_observed)
    S = compute_robust_covariance_from_standardized(
        X_observed_std,
        diagonal_loading=diagonal_loading,
    )

    A_path = get_path_adjacency(n_components)

    X_val_obs = X_observed[:n_val_metrics]
    X_val_obs_std = apply_observed_standardizer(X_val_obs, obs_mean, obs_std)
    X_val_orig = X_original[:n_val_metrics]

    orthogonal_glasso = should_use_orthogonal_alignment(output_representation)

    trial_results: Dict[str, Dict[str, Any]] = {}
    recovered_signals_by_method: Dict[str, np.ndarray] = {}

    # -------------------------------------------------------------------------
    # GLASSO grid
    # -------------------------------------------------------------------------
    if "glasso" in baselines:
        for alpha in alphas_to_test:
            method_key = f"glasso_alpha={alpha:g}"
            print(f"[Trial {trial_idx}] Testing {method_key}")

            try:
                W, frob_error = discover_glasso_topology_and_match_linear(
                    S,
                    A_path,
                    alpha,
                    orthogonal=orthogonal_glasso,
                )

                X_val_rec = X_val_obs_std @ W
                metrics = evaluate_recovery(X_val_orig, X_val_rec)

                trial_results[method_key] = {
                    "success": True,
                    "baseline": "glasso",
                    "alpha": float(alpha),
                    "frob_error": float(frob_error),
                    **metrics,
                }

                recovered_signals_by_method[method_key] = X_val_rec[:5]

                print(
                    f"  -> Success | corr={metrics['correlation_mean']:.4f} | "
                    f"sample std={metrics['correlation_std_across_samples']:.4f} | "
                    f"frob={frob_error:.4f}"
                )

            except Exception as e:
                trial_results[method_key] = {
                    "success": False,
                    "baseline": "glasso",
                    "alpha": float(alpha),
                    "error": str(e),
                }
                print(f"  -> Failed | {method_key} | error={e}")

    # -------------------------------------------------------------------------
    # Spectral covariance matching grid
    # -------------------------------------------------------------------------
    if "spectral" in baselines:
        for mode in spectral_modes:
            if mode in ["rotate", "whiten"]:
                configs = [(None, None)]
            elif mode == "recolor":
                configs = [
                    (gamma, power)
                    for gamma in spectral_gammas
                    for power in spectral_powers
                ]
            else:
                raise ValueError(f"Unknown spectral mode: {mode}")

            for gamma, power in configs:
                if mode in ["rotate", "whiten"]:
                    method_key = f"spectral_{mode}"
                    gamma_value = 1.0
                    power_value = 1.0
                else:
                    method_key = f"spectral_recolor_gamma={gamma:g}_power={power:g}"
                    gamma_value = float(gamma)
                    power_value = float(power)

                print(f"[Trial {trial_idx}] Testing {method_key}")

                try:
                    W, info = discover_spectral_covariance_matching(
                        S,
                        A_path,
                        mode=mode,
                        gamma=gamma_value,
                        power=power_value,
                    )

                    X_val_rec = X_val_obs_std @ W
                    metrics = evaluate_recovery(X_val_orig, X_val_rec)

                    trial_results[method_key] = {
                        "success": True,
                        "baseline": "spectral_covariance_matching",
                        **info,
                        **metrics,
                    }

                    recovered_signals_by_method[method_key] = X_val_rec[:5]

                    print(
                        f"  -> Success | corr={metrics['correlation_mean']:.4f} | "
                        f"sample std={metrics['correlation_std_across_samples']:.4f}"
                    )

                except Exception as e:
                    trial_results[method_key] = {
                        "success": False,
                        "baseline": "spectral_covariance_matching",
                        "mode": mode,
                        "gamma": None if gamma is None else float(gamma),
                        "power": None if power is None else float(power),
                        "error": str(e),
                    }
                    print(f"  -> Failed | {method_key} | error={e}")

    if save_plot and recovered_signals_by_method:
        plot_dir.mkdir(parents=True, exist_ok=True)

        successful_keys = [
            key
            for key, result in trial_results.items()
            if result.get("success", False)
        ]
        successful_keys = sorted(
            successful_keys,
            key=lambda key: trial_results[key]["correlation_mean"],
            reverse=True,
        )
        selected_keys = successful_keys[:n_plot_methods]

        selected_recoveries = {
            key: recovered_signals_by_method[key]
            for key in selected_keys
            if key in recovered_signals_by_method
        }

        plot_filename = (
            f"{feature}_{output_representation}_"
            f"trial_{trial_idx:03d}_seed_{seed}_baselines.png"
        )
        plot_path = plot_dir / plot_filename

        observed_label = (
            "Linear Mixture"
            if output_representation == "linear"
            else "Observed"
        )

        plot_all_recoveries_gsn(
            X_original=X_original[:5],
            X_observed=X_observed[:5],
            recovered_dict=selected_recoveries,
            N=n_components,
            n_plot=5,
            save_path=str(plot_path),
            observed_label=observed_label,
        )

        print(f"[Trial {trial_idx}] Saved recovery plot to {plot_path}")

    return trial_results


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_results(all_trial_results: Dict[str, Dict[str, Dict[str, Any]]]):
    summary: Dict[str, Dict[str, Any]] = {}

    all_method_keys = sorted(
        {
            method_key
            for trial_result in all_trial_results.values()
            for method_key in trial_result.keys()
        }
    )

    for method_key in all_method_keys:
        corr_means = []
        corr_sample_stds = []
        corr_sample_vars = []
        frob_errors = []
        condition_numbers = []
        failed_trials = []
        baseline_name = None
        method_metadata: Dict[str, Any] = {}

        for trial_key, trial_result in all_trial_results.items():
            result = trial_result.get(method_key)

            if result is None or not result.get("success", False):
                failed_trials.append(trial_key)
                continue

            baseline_name = result.get("baseline", baseline_name)
            corr_means.append(result["correlation_mean"])
            corr_sample_stds.append(result["correlation_std_across_samples"])
            corr_sample_vars.append(result["correlation_variance_across_samples"])

            if "frob_error" in result:
                frob_errors.append(result["frob_error"])
            if "empirical_condition_number" in result:
                condition_numbers.append(result["empirical_condition_number"])

            for key in ["baseline", "alpha", "mode", "gamma", "power"]:
                if key in result:
                    method_metadata[key] = result[key]

        corr_means = np.array(corr_means, dtype=float)
        corr_sample_stds = np.array(corr_sample_stds, dtype=float)
        corr_sample_vars = np.array(corr_sample_vars, dtype=float)
        frob_errors = np.array(frob_errors, dtype=float)
        condition_numbers = np.array(condition_numbers, dtype=float)

        n_successful = len(corr_means)

        if n_successful == 0:
            summary[method_key] = {
                "baseline": baseline_name,
                "n_successful_trials": 0,
                "failed_trials": failed_trials,
                **method_metadata,
            }
            continue

        stats = {
            "baseline": baseline_name,
            "n_successful_trials": int(n_successful),
            "failed_trials": failed_trials,
            "correlation_mean_across_trials": float(np.mean(corr_means)),
            "correlation_std_across_trials": float(np.std(corr_means, ddof=1))
            if n_successful > 1
            else 0.0,
            "correlation_variance_across_trials": float(np.var(corr_means, ddof=1))
            if n_successful > 1
            else 0.0,
            "mean_correlation_std_across_samples": float(np.mean(corr_sample_stds)),
            "mean_correlation_variance_across_samples": float(np.mean(corr_sample_vars)),
            **method_metadata,
        }

        if len(frob_errors) > 0:
            stats.update(
                {
                    "frob_error_mean_across_trials": float(np.mean(frob_errors)),
                    "frob_error_std_across_trials": float(np.std(frob_errors, ddof=1))
                    if len(frob_errors) > 1
                    else 0.0,
                    "frob_error_variance_across_trials": float(np.var(frob_errors, ddof=1))
                    if len(frob_errors) > 1
                    else 0.0,
                }
            )

        if len(condition_numbers) > 0:
            stats.update(
                {
                    "empirical_condition_number_mean_across_trials": float(
                        np.mean(condition_numbers)
                    ),
                    "empirical_condition_number_std_across_trials": float(
                        np.std(condition_numbers, ddof=1)
                    )
                    if len(condition_numbers) > 1
                    else 0.0,
                }
            )

        summary[method_key] = stats

    return summary


def get_best_summary_key(summary: Dict[str, Dict[str, Any]]) -> Optional[str]:
    successful_keys = [
        key
        for key, stats in summary.items()
        if stats.get("n_successful_trials", 0) > 0
    ]
    if not successful_keys:
        return None
    return max(
        successful_keys,
        key=lambda key: summary[key]["correlation_mean_across_trials"],
    )


def print_summary(summary: Dict[str, Dict[str, Any]], top_k: int = 20):
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    successful_items = [
        (method_key, stats)
        for method_key, stats in summary.items()
        if stats.get("n_successful_trials", 0) > 0
    ]

    failed_items = [
        (method_key, stats)
        for method_key, stats in summary.items()
        if stats.get("n_successful_trials", 0) == 0
    ]

    successful_items = sorted(
        successful_items,
        key=lambda item: item[1]["correlation_mean_across_trials"],
        reverse=True,
    )

    for rank, (method_key, stats) in enumerate(successful_items[:top_k], start=1):
        frob_text = ""
        if "frob_error_mean_across_trials" in stats:
            frob_text = (
                f" | frob={stats['frob_error_mean_across_trials']:.4f} "
                f"± {stats['frob_error_std_across_trials']:.4f}"
            )

        print(
            f"#{rank:02d} {method_key}: "
            f"corr={stats['correlation_mean_across_trials']:.4f} "
            f"± {stats['correlation_std_across_trials']:.4f} | "
            f"var={stats['correlation_variance_across_trials']:.6f}"
            f"{frob_text} | n={stats['n_successful_trials']}"
        )

    if failed_items:
        print("\nFailed configurations:")
        for method_key, _ in failed_items:
            print(f"  - {method_key}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated latent topology recovery experiments with GLASSO and "
            "Spectral Covariance Matching baselines."
        )
    )

    parser.add_argument(
        "--feature",
        type=str,
        default="gaussian",
        choices=["gaussian", "legendre", "ising"],
        help="Feature/data type to generate.",
    )

    parser.add_argument(
        "--output-representation",
        type=str,
        default="linear",
        choices=["natural", "dst", "permuted", "linear"],
        help="Observed output representation.",
    )

    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["glasso", "spectral"],
        choices=["glasso", "spectral"],
        help="Which baselines to run.",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Number of independent repeated trials.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Trial k uses seed + k.",
    )

    parser.add_argument(
        "--n-components",
        type=int,
        default=63,
        help="Number of latent components.",
    )

    parser.add_argument(
        "--n-total-samples",
        type=int,
        default=500000,
        help="Number of generated samples per trial.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Generation batch size.",
    )

    parser.add_argument(
        "--n-val-metrics",
        type=int,
        default=500,
        help="Number of validation samples used for correlation metrics.",
    )

    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24],
        help="List of GLASSO alpha values to test.",
    )

    parser.add_argument(
        "--spectral-modes",
        type=str,
        nargs="+",
        default=["rotate", "recolor"],
        choices=["rotate", "whiten", "recolor"],
        help="Spectral covariance matching variants to test.",
    )

    parser.add_argument(
        "--spectral-gammas",
        type=float,
        nargs="+",
        default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        help="Gamma values for the smooth path spectrum used by spectral_recolor.",
    )

    parser.add_argument(
        "--spectral-powers",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="Power values for the smooth path spectrum used by spectral_recolor.",
    )

    parser.add_argument(
        "--diagonal-loading",
        type=float,
        default=1e-2,
        help="Diagonal loading added to empirical covariance.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where JSON results and plots are saved.",
    )

    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save recovery plots for each trial.",
    )

    parser.add_argument(
        "--n-plot-methods",
        type=int,
        default=8,
        help="Number of best baseline configurations to include in recovery plots.",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = output_dir / "plots"

    baseline_name = "_".join(args.baselines)
    experiment_name = (
        f"{args.feature}_{args.output_representation}_"
        f"{baseline_name}_{args.n_trials}trials"
    )

    json_path = output_dir / f"{experiment_name}_results.json"

    print("=" * 100)
    print("Repeated Baseline Experiment")
    print("=" * 100)
    print(f"feature                 : {args.feature}")
    print(f"output_representation   : {args.output_representation}")
    print(f"baselines               : {args.baselines}")
    print(f"n_trials                : {args.n_trials}")
    print(f"base seed               : {args.seed}")
    print(f"GLASSO alphas           : {args.alphas}")
    print(f"spectral modes          : {args.spectral_modes}")
    print(f"spectral gammas         : {args.spectral_gammas}")
    print(f"spectral powers         : {args.spectral_powers}")
    print(f"results path            : {json_path}")
    print("=" * 100)

    all_trial_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for trial_idx in range(args.n_trials):
        seed = args.seed + trial_idx
        trial_key = f"trial_{trial_idx:03d}"

        trial_results = run_single_trial(
            trial_idx=trial_idx,
            seed=seed,
            feature=args.feature,
            output_representation=args.output_representation,
            n_components=args.n_components,
            n_total_samples=args.n_total_samples,
            batch_size=args.batch_size,
            n_val_metrics=args.n_val_metrics,
            baselines=args.baselines,
            alphas_to_test=args.alphas,
            spectral_modes=args.spectral_modes,
            spectral_gammas=args.spectral_gammas,
            spectral_powers=args.spectral_powers,
            diagonal_loading=args.diagonal_loading,
            save_plot=args.save_plots,
            plot_dir=plot_dir,
            n_plot_methods=args.n_plot_methods,
        )

        all_trial_results[trial_key] = trial_results

    summary = aggregate_results(all_trial_results=all_trial_results)
    best_key = get_best_summary_key(summary)

    output = {
        "config": {
            "feature": args.feature,
            "output_representation": args.output_representation,
            "baselines": args.baselines,
            "n_trials": args.n_trials,
            "base_seed": args.seed,
            "n_components": args.n_components,
            "n_total_samples": args.n_total_samples,
            "batch_size": args.batch_size,
            "n_val_metrics": args.n_val_metrics,
            "alphas": args.alphas,
            "spectral_modes": args.spectral_modes,
            "spectral_gammas": args.spectral_gammas,
            "spectral_powers": args.spectral_powers,
            "diagonal_loading": args.diagonal_loading,
        },
        "best_configuration": best_key,
        "summary": summary,
        "trials": all_trial_results,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=4)

    print_summary(summary)

    if best_key is not None:
        best_stats = summary[best_key]
        print("\nBest configuration:")
        print(
            f"  {best_key}: corr="
            f"{best_stats['correlation_mean_across_trials']:.4f} "
            f"± {best_stats['correlation_std_across_trials']:.4f}"
        )

    print("\nSaved full results to:")
    print(json_path)


if __name__ == "__main__":
    main()
