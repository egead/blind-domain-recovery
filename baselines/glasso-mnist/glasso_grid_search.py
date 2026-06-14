import argparse
import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import graphical_lasso
from scipy.optimize import quadratic_assignment

from synthetic_data_generator import DataGenerator2d


# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def to_numpy(x):
    """Converts TensorFlow tensors to NumPy arrays when needed."""
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


# =============================================================================
# 1. TOPOLOGY HELPERS
# =============================================================================

def get_grid_adjacency(H: int, W: int) -> np.ndarray:
    """Creates the exact adjacency matrix for an H x W 2D grid."""
    n_nodes = H * W
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            idx = i * W + j

            if i > 0:
                A[idx, (i - 1) * W + j] = 1

            if i < H - 1:
                A[idx, (i + 1) * W + j] = 1

            if j > 0:
                A[idx, i * W + (j - 1)] = 1

            if j < W - 1:
                A[idx, i * W + (j + 1)] = 1

    return A


def compute_robust_covariance(
    X: np.ndarray,
    diagonal_loading: float = 1e-2,
) -> np.ndarray:
    """Standardizes data and computes a robust SPD empirical covariance matrix."""
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    S = np.cov(X_std, rowvar=False)
    S = 0.5 * (S + S.T)

    np.fill_diagonal(S, S.diagonal() + diagonal_loading)

    return S


# =============================================================================
# 2. MATCHING ALGORITHMS
# =============================================================================

def discover_topology_faq(
    S: np.ndarray,
    A_target: np.ndarray,
    alpha: float,
):
    """Runs GLASSO and uses FAQ to find a strict permutation matrix P."""
    _, precision_ = graphical_lasso(
        emp_cov=S,
        alpha=alpha,
        max_iter=200,
    )

    A_glasso = np.abs(precision_)
    np.fill_diagonal(A_glasso, 0)

    A_glasso = (A_glasso > 1e-3).astype(np.float32)

    res = quadratic_assignment(
        A_glasso,
        A_target,
        options={"maximize": True},
    )

    p_indices = res.col_ind

    n = A_target.shape[0]
    P = np.zeros((n, n), dtype=np.float32)
    P[np.arange(n), p_indices] = 1.0

    frob_error = np.linalg.norm(
        A_glasso - P @ A_target @ P.T,
        ord="fro",
    )

    return P, float(frob_error)


def discover_topology_spectral(
    S: np.ndarray,
    A_target: np.ndarray,
    alpha: float,
    orthogonal: bool = True,
):
    """Runs GLASSO and uses eigendecomposition to find a dense unmixing matrix W."""
    _, precision_ = graphical_lasso(
        emp_cov=S,
        alpha=alpha,
        max_iter=200,
    )

    Theta_glasso = np.abs(precision_)
    np.fill_diagonal(Theta_glasso, 0)

    Theta_glasso = np.where(Theta_glasso > 1e-3, Theta_glasso, 0.0)

    vals_g, vecs_g = np.linalg.eigh(Theta_glasso)
    vals_t, vecs_t = np.linalg.eigh(A_target)

    idx_g = np.argsort(vals_g)[::-1]
    vals_g = vals_g[idx_g]
    vecs_g = vecs_g[:, idx_g]

    idx_t = np.argsort(vals_t)[::-1]
    vals_t = vals_t[idx_t]
    vecs_t = vecs_t[:, idx_t]

    if orthogonal:
        W = vecs_g @ vecs_t.T
    else:
        scale = np.sqrt(np.abs(vals_t) / (np.abs(vals_g) + 1e-8))
        W = vecs_g @ np.diag(scale) @ vecs_t.T

    frob_error = np.linalg.norm(
        Theta_glasso - W @ A_target @ W.T,
        ord="fro",
    )

    return W, float(frob_error)


# =============================================================================
# 3. METRIC
# =============================================================================

def compute_max_correlation_2d(
    img_orig: np.ndarray,
    img_rec: np.ndarray,
) -> float:
    """
    Computes max absolute Pearson correlation under natural 2D ambiguities:
    rotations, reflections, and sign inversion through abs(correlation).
    """
    orig_flat = img_orig.flatten()

    if np.std(orig_flat) < 1e-8:
        return 0.0

    candidates = []

    for flip in [False, True]:
        curr_img = np.fliplr(img_rec) if flip else img_rec

        for k in range(4):
            rotated_img = np.rot90(curr_img, k=k)

            if rotated_img.shape == img_orig.shape:
                candidates.append(rotated_img)

    max_corr = 0.0

    for candidate in candidates:
        rec_flat = candidate.flatten()

        if np.std(rec_flat) < 1e-8:
            continue

        corr = np.corrcoef(orig_flat, rec_flat)[0, 1]

        if np.isnan(corr):
            corr = 0.0

        max_corr = max(max_corr, abs(corr))

    return float(max_corr)


# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def plot_recoveries_2d(
    X_original,
    X_observed,
    recovered_dict,
    H,
    W,
    n_plot=5,
    save_path="recovery_2d.png",
):
    alphas = list(recovered_dict.keys())
    n_rows = 2 + len(alphas)

    fig, axes = plt.subplots(
        n_rows,
        n_plot,
        figsize=(12, 2 * n_rows),
    )

    if n_plot == 1:
        axes = axes[:, None]

    for col in range(n_plot):
        axes[0, col].imshow(
            X_original[col].reshape(H, W),
            cmap="gray",
        )
        axes[0, col].axis("off")

        if col == 0:
            axes[0, col].set_title("Original")

        axes[1, col].imshow(
            X_observed[col].reshape(H, W),
            cmap="gray",
        )
        axes[1, col].axis("off")

        if col == 0:
            axes[1, col].set_title("Observed")

        for row_idx, alpha in enumerate(alphas):
            axes[row_idx + 2, col].imshow(
                recovered_dict[alpha][col].reshape(H, W),
                cmap="gray",
            )
            axes[row_idx + 2, col].axis("off")

            if col == 0:
                axes[row_idx + 2, col].set_title(f"Recovered α={alpha}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# =============================================================================
# 5. EXPERIMENT SETUP
# =============================================================================

def resolve_output_representation(
    output_representation: str,
    matching_method: str,
) -> str:
    """
    If output_representation='auto':
    - FAQ uses 'permuted', because it returns a strict permutation.
    - Spectral uses 'linear', because it can handle dense linear mixing.
    """
    if output_representation != "auto":
        return output_representation

    if matching_method == "faq":
        return "permuted"

    return "linear"


def build_mnist_experiment(
    *,
    output_representation: str,
    batch_size: int,
    H: int,
    W: int,
):
    n_components = H * W

    A_target = get_grid_adjacency(H, W)

    generator = DataGenerator2d(
        batch_size=batch_size,
        features=[
            {
                "type": "mnist",
                "apply_padding": False,
            }
        ],
        latent_x_dims=H,
        latent_y_dims=W,
        output_representation=output_representation,
        flatten_output=True,
    )

    return generator, A_target, n_components


# =============================================================================
# 6. SINGLE TRIAL
# =============================================================================

def run_single_trial(
    *,
    trial_idx: int,
    seed: int,
    matching_method: str,
    output_representation: str,
    orthogonal: bool,
    n_total_samples: int,
    batch_size: int,
    n_val_metrics: int,
    alphas_to_test: list[float],
    diagonal_loading: float,
    H: int,
    W: int,
    save_plots: bool,
    plot_dir: Path,
):
    set_seed(seed)

    generator, A_target, n_components = build_mnist_experiment(
        output_representation=output_representation,
        batch_size=batch_size,
        H=H,
        W=W,
    )

    print(
        f"\n[Trial {trial_idx}] "
        f"dataset=mnist, "
        f"matching={matching_method}, "
        f"output={output_representation}, "
        f"orthogonal={orthogonal}, "
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

        X_obs_batch = to_numpy(X_obs_batch)
        X_orig_batch = to_numpy(X_orig_batch)

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

    X_observed = X_observed.reshape(n_total_samples, n_components)
    X_original_for_metrics = X_original.reshape(-1, H, W)

    print(f"[Trial {trial_idx}] Computing empirical covariance...")

    S = compute_robust_covariance(
        X_observed,
        diagonal_loading=diagonal_loading,
    )

    X_val_obs = X_observed[:n_val_metrics]
    X_val_orig = X_original_for_metrics[:n_val_metrics]

    trial_results = {}
    recovered_signals_by_alpha = {}

    print(f"[Trial {trial_idx}] Starting alpha sweep...")

    for alpha in alphas_to_test:
        alpha_key = str(alpha)

        try:
            if matching_method == "faq":
                M, frob_error = discover_topology_faq(
                    S,
                    A_target,
                    alpha,
                )

            elif matching_method == "spectral":
                M, frob_error = discover_topology_spectral(
                    S,
                    A_target,
                    alpha,
                    orthogonal=orthogonal,
                )

            else:
                raise ValueError(f"Unknown matching method: {matching_method}")

            X_val_rec_flat = X_val_obs @ M
            X_val_rec = X_val_rec_flat.reshape(-1, H, W)

            correlations = [
                compute_max_correlation_2d(
                    X_val_orig[i],
                    X_val_rec[i],
                )
                for i in range(n_val_metrics)
            ]

            correlations = np.array(correlations, dtype=float)

            trial_results[alpha_key] = {
                "success": True,
                "frob_error": float(frob_error),
                "correlation_mean": float(np.mean(correlations)),
                "correlation_std_across_samples": float(
                    np.std(correlations, ddof=1)
                ),
                "correlation_variance_across_samples": float(
                    np.var(correlations, ddof=1)
                ),
                "correlation_min": float(np.min(correlations)),
                "correlation_max": float(np.max(correlations)),
            }

            recovered_signals_by_alpha[alpha] = X_val_rec_flat[:5]

            print(
                f"  alpha={alpha:<8} | "
                f"corr={trial_results[alpha_key]['correlation_mean']:.4f} | "
                f"sample std={trial_results[alpha_key]['correlation_std_across_samples']:.4f} | "
                f"frob={frob_error:.4f}"
            )

        except Exception as e:
            trial_results[alpha_key] = {
                "success": False,
                "error": str(e),
            }

            print(f"  alpha={alpha:<8} | Failed: {e}")

    if save_plots and recovered_signals_by_alpha:
        plot_dir.mkdir(parents=True, exist_ok=True)

        plot_path = plot_dir / (
            f"mnist_{matching_method}_{output_representation}_"
            f"trial_{trial_idx:03d}_seed_{seed}.png"
        )

        plot_recoveries_2d(
            X_original=X_original_for_metrics[:5],
            X_observed=X_observed[:5],
            recovered_dict=recovered_signals_by_alpha,
            H=H,
            W=W,
            n_plot=5,
            save_path=str(plot_path),
        )

        print(f"[Trial {trial_idx}] Saved plot to {plot_path}")

    return trial_results


# =============================================================================
# 7. AGGREGATION
# =============================================================================

def aggregate_results(
    all_trial_results: dict,
    alphas_to_test: list[float],
):
    summary = {}

    for alpha in alphas_to_test:
        alpha_key = str(alpha)

        corr_means = []
        corr_sample_stds = []
        corr_sample_vars = []
        frob_errors = []
        failed_trials = []

        for trial_key, trial_result in all_trial_results.items():
            result = trial_result.get(alpha_key)

            if result is None or not result.get("success", False):
                failed_trials.append(trial_key)
                continue

            corr_means.append(result["correlation_mean"])
            corr_sample_stds.append(result["correlation_std_across_samples"])
            corr_sample_vars.append(result["correlation_variance_across_samples"])
            frob_errors.append(result["frob_error"])

        corr_means = np.array(corr_means, dtype=float)
        corr_sample_stds = np.array(corr_sample_stds, dtype=float)
        corr_sample_vars = np.array(corr_sample_vars, dtype=float)
        frob_errors = np.array(frob_errors, dtype=float)

        n_successful = len(corr_means)

        if n_successful == 0:
            summary[alpha_key] = {
                "n_successful_trials": 0,
                "failed_trials": failed_trials,
            }
            continue

        summary[alpha_key] = {
            "n_successful_trials": int(n_successful),
            "failed_trials": failed_trials,

            "correlation_mean_across_trials": float(np.mean(corr_means)),
            "correlation_std_across_trials": float(
                np.std(corr_means, ddof=1)
            ) if n_successful > 1 else 0.0,
            "correlation_variance_across_trials": float(
                np.var(corr_means, ddof=1)
            ) if n_successful > 1 else 0.0,

            "mean_correlation_std_across_samples": float(
                np.mean(corr_sample_stds)
            ),
            "mean_correlation_variance_across_samples": float(
                np.mean(corr_sample_vars)
            ),

            "frob_error_mean_across_trials": float(np.mean(frob_errors)),
            "frob_error_std_across_trials": float(
                np.std(frob_errors, ddof=1)
            ) if n_successful > 1 else 0.0,
            "frob_error_variance_across_trials": float(
                np.var(frob_errors, ddof=1)
            ) if n_successful > 1 else 0.0,
        }

    return summary


def print_summary(summary: dict):
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for alpha_key, stats in summary.items():
        if stats["n_successful_trials"] == 0:
            print(f"alpha={alpha_key}: all trials failed")
            continue

        print(
            f"alpha={alpha_key:<8} | "
            f"corr={stats['correlation_mean_across_trials']:.4f} "
            f"± {stats['correlation_std_across_trials']:.4f} | "
            f"var={stats['correlation_variance_across_trials']:.6f} | "
            f"frob={stats['frob_error_mean_across_trials']:.4f} "
            f"± {stats['frob_error_std_across_trials']:.4f} | "
            f"n={stats['n_successful_trials']}"
        )


# =============================================================================
# 8. CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Repeated GLASSO topology recovery experiments for MNIST."
    )

    parser.add_argument(
        "--matching-method",
        type=str,
        default="spectral",
        choices=["faq", "spectral"],
        help="Topology matching method.",
    )

    parser.add_argument(
        "--output-representation",
        type=str,
        default="auto",
        choices=["auto", "permuted", "linear"],
        help=(
            "Observed output representation. "
            "Use 'auto' to choose 'permuted' for FAQ and 'linear' for spectral."
        ),
    )

    parser.set_defaults(orthogonal=True)

    parser.add_argument(
        "--orthogonal",
        action="store_true",
        dest="orthogonal",
        help="Use orthogonal spectral alignment. This is the default.",
    )

    parser.add_argument(
        "--non-orthogonal",
        action="store_false",
        dest="orthogonal",
        help="Use non-orthogonal spectral alignment with eigenvalue scaling.",
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
        help="Base seed. Trial k uses seed + k.",
    )

    parser.add_argument(
        "--n-total-samples",
        type=int,
        default=50000,
        help="Number of generated samples per trial.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for data generation.",
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
        default=[0.01, 0.05, 0.25, 1.25, 6.25],
        help="GLASSO alpha values to test.",
    )

    parser.add_argument(
        "--diagonal-loading",
        type=float,
        default=1e-2,
        help="Diagonal loading added to covariance matrix.",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=15,
        help="MNIST latent grid height.",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=15,
        help="MNIST latent grid width.",
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

    return parser.parse_args()


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    args = parse_args()

    output_representation = resolve_output_representation(
        output_representation=args.output_representation,
        matching_method=args.matching_method,
    )

    if args.matching_method == "faq" and output_representation != "permuted":
        print(
            "\n[Warning] FAQ returns a strict permutation matrix. "
            "It is usually appropriate for output_representation='permuted'. "
            f"You selected output_representation='{output_representation}'.\n"
        )

    if args.matching_method == "spectral" and output_representation == "permuted":
        print(
            "\n[Warning] Spectral matching can run with 'permuted', but "
            "the default intended stress test is output_representation='linear'.\n"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = output_dir / "plots"

    experiment_name = (
        f"mnist_{args.matching_method}_{output_representation}_"
        f"{args.n_trials}trials"
    )

    json_path = output_dir / f"{experiment_name}_results.json"

    print("=" * 80)
    print("GLASSO Repeated MNIST Experiment")
    print("=" * 80)
    print(f"dataset               : mnist")
    print(f"matching_method       : {args.matching_method}")
    print(f"output_representation : {output_representation}")
    print(f"orthogonal            : {args.orthogonal}")
    print(f"n_trials              : {args.n_trials}")
    print(f"base_seed             : {args.seed}")
    print(f"grid size             : {args.H} x {args.W}")
    print(f"alphas                : {args.alphas}")
    print(f"results_path          : {json_path}")
    print("=" * 80)

    all_trial_results = {}

    for trial_idx in range(args.n_trials):
        seed = args.seed + trial_idx
        trial_key = f"trial_{trial_idx:03d}"

        trial_results = run_single_trial(
            trial_idx=trial_idx,
            seed=seed,
            matching_method=args.matching_method,
            output_representation=output_representation,
            orthogonal=args.orthogonal,
            n_total_samples=args.n_total_samples,
            batch_size=args.batch_size,
            n_val_metrics=args.n_val_metrics,
            alphas_to_test=args.alphas,
            diagonal_loading=args.diagonal_loading,
            H=args.H,
            W=args.W,
            save_plots=args.save_plots,
            plot_dir=plot_dir,
        )

        all_trial_results[trial_key] = trial_results

    summary = aggregate_results(
        all_trial_results=all_trial_results,
        alphas_to_test=args.alphas,
    )

    output = {
        "config": {
            "dataset": "mnist",
            "matching_method": args.matching_method,
            "output_representation": output_representation,
            "orthogonal": args.orthogonal,
            "n_trials": args.n_trials,
            "base_seed": args.seed,
            "n_total_samples": args.n_total_samples,
            "batch_size": args.batch_size,
            "n_val_metrics": args.n_val_metrics,
            "alphas": args.alphas,
            "diagonal_loading": args.diagonal_loading,
            "H": args.H,
            "W": args.W,
        },
        "summary": summary,
        "trials": all_trial_results,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=4)

    print_summary(summary)

    print("\nSaved full results to:")
    print(json_path)


if __name__ == "__main__":
    main()