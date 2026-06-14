import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import json # Added to store results

# Import your data generator
from synthetic_data_generator import DataGenerator2d

# Import the refactored topology classes
from topology_learner import TopologyConfig, TopologyLearner, ImageReconstructor

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
    
    # Protect against blank images causing division by zero
    if std_orig < 1e-8:
        return 0.0
        
    max_corr = -1.0
    
    # Test all 8 variations (4 rotations * 2 flips)
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
# Visualization
# =============================================================================
def plot_all_recoveries(X_original: np.ndarray, 
                        X_shuffled: np.ndarray, 
                        recovered_dict: dict, 
                        H: int, W: int, 
                        n_plot: int = 5):
    """Plots original, shuffled, and the recovered images for all grid search combinations."""
    hyperparams = list(recovered_dict.keys())
    n_rows = 2 + len(hyperparams)
    
    fig, axes = plt.subplots(n_rows, n_plot, figsize=(12, 2.5 * n_rows))
    
    for col in range(n_plot):
        # 1. Original
        ax_orig = axes[0, col]
        ax_orig.imshow(X_original[col], cmap='gray')
        ax_orig.axis('off')
        if col == 0: 
            ax_orig.set_title("Original", fontweight='bold')
        
        # 2. Shuffled 
        ax_shuf = axes[1, col]
        ax_shuf.imshow(X_shuffled[col].reshape(H, W), cmap='gray')
        ax_shuf.axis('off')
        if col == 0: 
            ax_shuf.set_title("Shuffled (1D Array)", fontweight='bold')
        
        # 3. Recoveries per hyperparameter combination (vt, k)
        for row_idx, (vt, k) in enumerate(hyperparams):
            ax_rec = axes[row_idx + 2, col]
            
            # Extract the images and the computed correlation score
            X_rec = recovered_dict[(vt, k)]["images"][col] 
            score = recovered_dict[(vt, k)]["score"]
            
            ax_rec.imshow(X_rec, cmap='gray')
            ax_rec.axis('off')
            if col == 0: 
                ax_rec.set_title(f"vt={vt}, k={k}\nCorr: {score:.3f}", fontsize=11)

    plt.tight_layout()
    plt.savefig("manifold_grid_search_recovery.png", dpi=150)
    print("\nPlot saved as 'manifold_grid_search_recovery.png'.")

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
    N_TOTAL_SAMPLES = 50000 
    BATCH_SIZE = 5000 
    N_TEST_METRICS = 100 # Number of samples to reconstruct for calculating average correlation
    
    # Grid Search Parameters
    METHOD = 'umap' # Can be 'isomap' or 'umap'
    FIXED_SMOOTHING_COEFF = 3.0
    VARIANCE_THRESHOLDS = [0.05, 0.10, 0.15]
    NEIGHBORS = [3, 5, 7] # Must be integers for ISOMAP/UMAP neighborhood sizing
    
    # --- 1. Data Generation ---
    generator = DataGenerator2d(
        batch_size=BATCH_SIZE,
        features=[{"type": "mnist", "apply_padding": False}],
        latent_x_dims=H,
        latent_y_dims=W,
        output_representation="permuted",
        flatten_output=True
    )

    print(f"Generating {N_TOTAL_SAMPLES} samples...")
    X_shuffled_list = []
    X_original_list = []
    samples_generated = 0

    while samples_generated < N_TOTAL_SAMPLES:
        X_shuf_batch, X_orig_batch = generator.sample_batch_of_data(return_hidden_signal=True)
        X_shuffled_list.append(X_shuf_batch.numpy())
        X_original_list.append(X_orig_batch.numpy())
        samples_generated += len(X_shuf_batch)
        print(f"Generated {samples_generated} / {N_TOTAL_SAMPLES}", end='\r')
    
    print("\nData generation complete.")

    X_shuffled = np.vstack(X_shuffled_list)[:N_TOTAL_SAMPLES]
    X_original = np.vstack(X_original_list)[:N_TOTAL_SAMPLES]
    X_original_flat = X_original.reshape(-1, H, W)

    # --- 2. 2D Grid Search ---
    print(f"\nStarting {METHOD.upper()} Grid Search (Fixed smoothing={FIXED_SMOOTHING_COEFF})...")
    recovered_images_dict = {}
    grid_search_results = {} # Dictionary to hold numeric results

    for vt, k in itertools.product(VARIANCE_THRESHOLDS, NEIGHBORS):
        print(f"--- Testing var_thresh = {vt}, k_neighbors = {k} ---")
        try:
            config = TopologyConfig(
                method=METHOD,
                variance_threshold=vt, 
                n_neighbors=k,
                grid_height=H,
                grid_width=W,
                smoothing_coefficient=FIXED_SMOOTHING_COEFF
            )
            
            learner = TopologyLearner(config)
            embeddings, weights, valid_idx = learner.recover_topology(X_shuffled)
            
            # Reconstruct a test batch and compute alignment correlations
            correlations = []
            X_rec_sample = []
            
            for i in range(N_TEST_METRICS):
                reconstructed = ImageReconstructor.reconstruct(X_shuffled[i], weights)
                
                # Save the first 5 for plotting
                if i < 5:
                    X_rec_sample.append(reconstructed)
                    
                # Calculate metric
                corr = compute_max_alignment_correlation(X_original_flat[i], reconstructed)
                correlations.append(corr)
                
            avg_corr = np.mean(correlations)
            print(f"  -> Topology learned successfully. Avg Max Correlation: {avg_corr:.4f}")
            
            # Store numeric metrics for JSON export
            config_key = f"vt_{vt}_k_{k}"
            grid_search_results[config_key] = {
                "variance_threshold": float(vt),
                "n_neighbors": int(k),
                "correlation": float(avg_corr)
            }
                
            # Store results mapped to hyperparameter tuple for plotting
            recovered_images_dict[(vt, k)] = {
                "images": X_rec_sample,
                "score": avg_corr
            }
            
        except Exception as e:
            print(f"  -> Failed with error: {e}")

    # Save results to a file
    if grid_search_results:
        results_filename = f"manifold_grid_search_results_{METHOD}.json"
        with open(results_filename, "w") as f:
            json.dump(grid_search_results, f, indent=4)
        print(f"\n[✔] Saved grid search numerical results to '{results_filename}'")

    # --- 3. Plotting ---
    if recovered_images_dict:
        print("\nPlotting results for successful hyperparameter combinations...")
        plot_all_recoveries(
            X_original=X_original_flat[:5], 
            X_shuffled=X_shuffled[:5], 
            recovered_dict=recovered_images_dict, 
            H=H, W=W, 
            n_plot=5
        )
    else:
        print("\nNo configurations converged successfully.")