import numpy as np
import tensorflow as tf
import json 
import os   
from sklearn.covariance import graphical_lasso
from scipy.optimize import quadratic_assignment

# Import your data generator
from synthetic_data_generator import DataGenerator2d

# =============================================================================
# Mathematical Helpers for GLASSO & Matching
# =============================================================================
def get_grid_adjacency(H: int, W: int) -> np.ndarray:
    """Creates the exact adjacency matrix for an H x W 2D grid."""
    n_nodes = H * W
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if i > 0: A[idx, (i - 1) * W + j] = 1       # Top
            if i < H - 1: A[idx, (i + 1) * W + j] = 1   # Bottom
            if j > 0: A[idx, i * W + (j - 1)] = 1       # Left
            if j < W - 1: A[idx, i * W + (j + 1)] = 1   # Right
            
    return A

def discover_topology_and_match_faq(S: np.ndarray, A_grid: np.ndarray, alpha: float):
    """
    Runs GLASSO and uses the FAQ algorithm (Quadratic Assignment) 
    to find the permutation matrix P.
    """
    _, precision_ = graphical_lasso(emp_cov=S, alpha=alpha, max_iter=200)
    
    A_glasso = np.abs(precision_)
    np.fill_diagonal(A_glasso, 0)
    A_glasso = (A_glasso > 1e-3).astype(np.float32)
    
    res = quadratic_assignment(A_glasso, A_grid, options={'maximize': True})
    p_indices = res.col_ind
    
    n = A_grid.shape[0]
    P = np.zeros((n, n))
    P[np.arange(n), p_indices] = 1.0
    
    frob_error = np.linalg.norm(A_glasso - P @ A_grid @ P.T, ord='fro')
    
    return P, frob_error

def discover_topology_and_match_spectral(S: np.ndarray, A_target: np.ndarray, alpha: float, orthogonal: bool = True):
    """
    Runs GLASSO and uses Eigendecomposition to find a dense unmixing matrix W.
    Works for any target adjacency graph (1D chain, 2D grid, etc.).
    """
    _, precision_ = graphical_lasso(emp_cov=S, alpha=alpha, max_iter=200)
    
    Theta_glasso = np.abs(precision_)
    np.fill_diagonal(Theta_glasso, 0)
    Theta_glasso = np.where(Theta_glasso > 1e-3, Theta_glasso, 0.0)
    
    # Eigendecomposition
    vals_g, vecs_g = np.linalg.eigh(Theta_glasso)
    vals_t, vecs_t = np.linalg.eigh(A_target)
    
    # Sort eigenvalues and eigenvectors descending
    idx_g = np.argsort(vals_g)[::-1]
    vecs_g, vals_g = vecs_g[:, idx_g], vals_g[idx_g]
    
    idx_t = np.argsort(vals_t)[::-1]
    vecs_t, vals_t = vecs_t[:, idx_t], vals_t[idx_t]
    
    # Construct unmixing matrix
    if orthogonal:
        W = vecs_g @ vecs_t.T
    else:
        scale = np.sqrt(np.abs(vals_t) / (np.abs(vals_g) + 1e-8))
        W = vecs_g @ np.diag(scale) @ vecs_t.T
    
    frob_error = np.linalg.norm(Theta_glasso - W @ A_target @ W.T, ord='fro')
    return W, frob_error

# =============================================================================
# Metrics
# =============================================================================
def compute_max_alignment_correlation(img_orig: np.ndarray, img_rec: np.ndarray) -> float:
    """
    Computes the maximum Pearson correlation between the original image 
    and all 8 dihedral rigid transformations of the recovered image.
    """
    orig_flat = img_orig.flatten()
    std_orig = np.std(orig_flat)
    
    if std_orig < 1e-8:
        return 0.0
        
    max_corr = -1.0
    for flip in [False, True]:
        curr_img = np.fliplr(img_rec) if flip else img_rec
        for k in range(4):
            rotated_img = np.rot90(curr_img, k=k)
            rec_flat = rotated_img.flatten()
            
            std_rec = np.std(rec_flat)
            if std_rec < 1e-8:
                corr = 0.0
            else:
                corr = np.corrcoef(orig_flat, rec_flat)[0, 1]
            if corr > max_corr:
                max_corr = np.abs(corr)
                
    return float(max_corr)

# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)

    # --- Configuration ---
    H, W = 15, 15
    D = H * W
    N_TOTAL_SAMPLES = 250000 
    BATCH_SIZE = 5000 
    N_TEST_METRICS = 500  
    DIAGONAL_LOADING = 1e-2
    N_TRIALS = 5  
    
    # --- EXPERIMENT CONFIGURATION FLAG ---
    MATCHING_METHOD = "spectral"  # Options: "faq" or "spectral"
    ORTHOGONAL_RECOVERY = True    # Options: True or False (Only applies to "spectral")
    
    # Determine the expected results file name based on configuration
    if MATCHING_METHOD == "faq":
        RESULTS_FILE = "mnist_faq_results.json"
        rep_mode = "permuted"
    else:
        RESULTS_FILE = "mnist_spectral_results.json"
        rep_mode = "linear"

    # --- Load Best Alpha dynamically (by highest correlation) ---
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            search_results = json.load(f)
            
        best_alpha_str = max(search_results, key=lambda k: search_results[k]["correlation"])
        BEST_ALPHA = float(best_alpha_str)
        best_corr = search_results[best_alpha_str]["correlation"]
        
        print(f"[✔] Loaded grid search results from '{RESULTS_FILE}'. Chosen BEST_ALPHA: {BEST_ALPHA} (Val Corr: {best_corr:.4f})")
    else:
        BEST_ALPHA = 1.25 # Fallback
        print(f"[!] Warning: '{RESULTS_FILE}' not found. Using fallback BEST_ALPHA: {BEST_ALPHA}")
    
    print(f"\n--- Characterizing Baseline on Streamed Dataset ---")
    print(f"Method: {MATCHING_METHOD.upper()} | Config: alpha={BEST_ALPHA}, diag_load={DIAGONAL_LOADING}")
    
    trial_scores = []
    trial_frob_errors = []

    for trial in range(N_TRIALS):
        trial_seed = 42 + trial
        np.random.seed(trial_seed)
        tf.random.set_seed(trial_seed)
        
        print(f"\nTrial {trial + 1}/{N_TRIALS} (Seed: {trial_seed})")
        
        generator = DataGenerator2d(
            seed=trial_seed,
            batch_size=BATCH_SIZE,
            features=[{"type": "mnist", "apply_padding": False}],
            latent_x_dims=H,
            latent_y_dims=W,
            output_representation=rep_mode, # Dynamically set based on matching method
            flatten_output=True
        )

        # 1. Online Training Phase (Streaming Covariance)
        print("  -> Streaming training batches...")
        sum_x = np.zeros(D, dtype=np.float64)
        sum_xx = np.zeros((D, D), dtype=np.float64)
        samples_generated = 0

        while samples_generated < N_TOTAL_SAMPLES:
            X_shuf_batch = generator.sample_batch_of_data(return_hidden_signal=False).numpy()
            
            X_batch_float64 = X_shuf_batch.astype(np.float64)
            sum_x += np.sum(X_batch_float64, axis=0)
            sum_xx += X_batch_float64.T @ X_batch_float64
            
            samples_generated += len(X_shuf_batch)
            print(f"     Processed {samples_generated} / {N_TOTAL_SAMPLES}", end='\r')

        # Compute final Covariance and Correlation matrices
        mean_x = sum_x / N_TOTAL_SAMPLES
        cov_matrix = (sum_xx / N_TOTAL_SAMPLES) - np.outer(mean_x, mean_x)
        
        stds = np.sqrt(np.clip(np.diag(cov_matrix), a_min=1e-12, a_max=None))
        corr_matrix = cov_matrix / np.outer(stds, stds)
        
        S = corr_matrix.copy()
        np.fill_diagonal(S, S.diagonal() + DIAGONAL_LOADING)
        
        # 2. Fit GLASSO & Match
        print(f"\n  -> Fitting GLASSO and solving {MATCHING_METHOD.upper()}...")
        A_grid = get_grid_adjacency(H, W)
        try:
            if MATCHING_METHOD == "faq":
                M, frob_error = discover_topology_and_match_faq(S, A_grid, BEST_ALPHA)
            else:
                M, frob_error = discover_topology_and_match_spectral(S, A_grid, BEST_ALPHA, orthogonal=ORTHOGONAL_RECOVERY)
                
            trial_frob_errors.append(frob_error)
            print(f"  -> Graph Matching Error (Frob Norm): {frob_error:.4f}")
        except Exception as e:
            print(f"  -> GLASSO Failed with error: {e}")
            continue
        
        # 3. Testing Phase
        print("  -> Generating test set for evaluation...")
        generator.batch_size = N_TEST_METRICS
        X_test_shuf, X_test_orig = generator.sample_batch_of_data(return_hidden_signal=True)
        
        X_test_shuf = X_test_shuf.numpy()
        X_test_orig = X_test_orig.numpy().reshape(-1, H, W)
        
        # Unmix the images using the learned transformation matrix (P or W)
        X_test_rec_flat = X_test_shuf @ M
        X_test_rec = X_test_rec_flat.reshape(-1, H, W)
        
        correlations = []
        for i in range(N_TEST_METRICS):
            corr = compute_max_alignment_correlation(X_test_orig[i], X_test_rec[i])
            correlations.append(corr)
            
        trial_avg = np.mean(correlations)
        trial_scores.append(trial_avg)
        
        print(f"  -> Trial Score (Avg Max Correlation): {trial_avg:.4f}")

    # --- Final Statistical Summary ---
    trial_scores = np.array(trial_scores)
    trial_frob_errors = np.array(trial_frob_errors)
    
    if len(trial_scores) > 0:
        mean_score = np.mean(trial_scores)
        std_score = np.std(trial_scores)
        
        mean_frob = np.mean(trial_frob_errors)
        std_frob = np.std(trial_frob_errors)
        
        print("\n==================================================")
        print("FINAL CHARACTERIZATION RESULTS")
        print("==================================================")
        print(f"Method:        GLASSO + {MATCHING_METHOD.upper()} (2D MNIST)")
        if MATCHING_METHOD == "spectral":
            print(f"Setting:       {'Orthogonal' if ORTHOGONAL_RECOVERY else 'General Linear'}")
        print(f"Trials:        {len(trial_scores)}/{N_TRIALS}")
        print(f"Test Set Size: {N_TEST_METRICS} images per trial")
        print(f"--------------------------------------------------")
        print(f"Graph Errors:  {np.round(trial_frob_errors, 4)}")
        print(f"Avg Frob Err:  {mean_frob:.4f} ± {std_frob:.4f}")
        print(f"--------------------------------------------------")
        print(f"Image Scores:  {np.round(trial_scores, 4)}")
        print(f"Robustness:    {mean_score:.4f} ± {std_score:.4f}")
        print("==================================================")
    else:
        print("\n[!] All trials failed. No statistics to summarize.")