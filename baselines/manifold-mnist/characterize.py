import numpy as np
import tensorflow as tf
import json # Added to read results
import os   # Added to check for file existence
from dataclasses import dataclass

# Import your data generator
from synthetic_data_generator import DataGenerator2d

# Import the refactored topology classes
from topology_learner import TopologyConfig, TopologyLearner, ImageReconstructor
import umap
from sklearn.manifold import Isomap

# =============================================================================
# Seedable Overrides
# =============================================================================
@dataclass
class SeededTopologyConfig(TopologyConfig):
    """Extends the config to accept a random state."""
    random_state: int = 42

class SeededTopologyLearner(TopologyLearner):
    """Overrides the embedding step to enforce the random seed."""
    def _compute_2d_embeddings(self, distance_matrix: np.ndarray) -> np.ndarray:
        method = self.config.method.lower()
        
        if method == 'isomap':
            # ISOMAP's internal PCA relies on global numpy random states
            np.random.seed(self.config.random_state)
            embedder = Isomap(
                n_neighbors=self.config.n_neighbors, 
                n_components=2, 
                metric='precomputed'
            )
        elif method == 'umap':
            # UMAP natively accepts a random_state
            embedder = umap.UMAP(
                n_neighbors=self.config.n_neighbors, 
                n_components=2, 
                metric='precomputed', 
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unsupported manifold learning method: {method}")
            
        return embedder.fit_transform(distance_matrix)

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

    # --- Configuration Parameters ---
    H, W = 15, 15
    N_TOTAL_SAMPLES = 250000 
    BATCH_SIZE = 2500 
    N_TEST_METRICS = 500  # Evaluated on more samples for a robust mean 
    
    METHOD = 'umap' 
    BEST_SMOOTHING = 3.0
    N_TRIALS = 5  # Number of independent runs
    
    # --- Load Best Hyperparameters dynamically ---
    RESULTS_FILE = f"manifold_grid_search_results_{METHOD}.json"
    
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            search_results = json.load(f)
            
        # Find the configuration key with the highest correlation
        best_key = max(search_results, key=lambda k: search_results[k]["correlation"])
        
        BEST_VAR_THRESH = search_results[best_key]["variance_threshold"]
        BEST_K_NEIGHBORS = search_results[best_key]["n_neighbors"]
        best_corr = search_results[best_key]["correlation"]
        
        print(f"[✔] Loaded grid search results.")
        print(f"    Chosen config: vt={BEST_VAR_THRESH}, k={BEST_K_NEIGHBORS}")
        print(f"    (Grid Search Validation Correlation: {best_corr:.4f})\n")
    else:
        # Fallback values if the first script was never run
        BEST_VAR_THRESH = 0.10
        BEST_K_NEIGHBORS = 3
        print(f"[!] Warning: '{RESULTS_FILE}' not found. Using fallback configuration.\n")
    
    print(f"--- Characterizing {METHOD.upper()} Robustness ---")
    print(f"Config: var_thresh={BEST_VAR_THRESH}, k={BEST_K_NEIGHBORS}, smoothing={BEST_SMOOTHING}")
    print(f"Running {N_TRIALS} independent seeded trials...\n")
    
    trial_scores = []

    for trial in range(N_TRIALS):
        # 1. Enforce Seeds for this Trial
        trial_seed = 42 + trial
        np.random.seed(trial_seed)
        tf.random.set_seed(trial_seed)
        
        print(f"Trial {trial + 1}/{N_TRIALS} (Seed: {trial_seed})")
        
        # 2. Generate Data
        generator = DataGenerator2d(
            seed=trial_seed,
            batch_size=BATCH_SIZE,
            features=[{"type": "mnist", "apply_padding": False}],
            latent_x_dims=H,
            latent_y_dims=W,
            output_representation="permuted",
            flatten_output=True
        )

        X_shuffled_list = []
        X_original_list = []
        samples_generated = 0

        while samples_generated < N_TOTAL_SAMPLES:
            X_shuf_batch, X_orig_batch = generator.sample_batch_of_data(return_hidden_signal=True)
            X_shuffled_list.append(X_shuf_batch.numpy())
            X_original_list.append(X_orig_batch.numpy())
            samples_generated += len(X_shuf_batch)

        X_shuffled = np.vstack(X_shuffled_list)[:N_TOTAL_SAMPLES]
        X_original = np.vstack(X_original_list)[:N_TOTAL_SAMPLES]
        X_original_flat = X_original.reshape(-1, H, W)

        # 3. Learn Topology
        config = SeededTopologyConfig(
            method=METHOD,
            variance_threshold=BEST_VAR_THRESH, 
            n_neighbors=BEST_K_NEIGHBORS,
            grid_height=H,
            grid_width=W,
            smoothing_coefficient=BEST_SMOOTHING,
            random_state=trial_seed
        )
        
        learner = SeededTopologyLearner(config)
        embeddings, weights, valid_idx = learner.recover_topology(X_shuffled)
        
        # 4. Evaluate Test Set
        correlations = []
        for i in range(N_TEST_METRICS):
            reconstructed = ImageReconstructor.reconstruct(X_shuffled[i], weights)
            corr = compute_max_alignment_correlation(X_original_flat[i], reconstructed)
            correlations.append(corr)
            
        trial_avg = np.mean(correlations)
        trial_scores.append(trial_avg)
        
        print(f"  -> Trial Score (Avg Max Correlation): {trial_avg:.4f}\n")

    # --- Final Statistical Summary ---
    trial_scores = np.array(trial_scores)
    mean_score = np.mean(trial_scores)
    std_score = np.std(trial_scores)
    
    print("==================================================")
    print("FINAL CHARACTERIZATION RESULTS")
    print("==================================================")
    print(f"Method:        {METHOD.upper()}")
    print(f"Trials:        {N_TRIALS}")
    print(f"Test Set Size: {N_TEST_METRICS} images per trial")
    print(f"Scores:        {np.round(trial_scores, 4)}")
    print(f"Robustness:    {mean_score:.4f} ± {std_score:.4f}")
    print("==================================================")