import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from synthetic_data_generator import DataGenerator
from neural_mapper import NeuralMapper
from scipy.stats import circmean

# =============================================================================
# CONFIGURATION & HYPERPARAMETERS
# =============================================================================
SESSION_ID = 732592105  # Example Allen Session
BATCH_SIZE = 1000
N_NEURONS = 110         # Number of top-variance neurons to use
CACHE_DIR = "../../allen_data"

# Grid Search Space
METHODS = ['umap', 'isomap']
N_NEIGHBORS_LIST = [5, 15, 30, 50]
MIN_DISTS = [0.01, 0.1, 0.5]  # Only applies to UMAP

# ==============================================================================
# 1. CORE MATH FUNCTIONS
# ==============================================================================

def circular_correlation(alpha, beta):
    """
    Computes the Circular Correlation Coefficient (Fisher & Lee, 1983).
    Range: [-1, 1].
    alpha, beta: Arrays in Radians [0, 2*pi].
    """
    # Compute circular means
    mean_alpha = circmean(alpha)
    mean_beta = circmean(beta)
    
    # Compute sin differences from mean
    a_diff = np.sin(alpha - mean_alpha)
    b_diff = np.sin(beta - mean_beta)
    
    num = np.sum(a_diff * b_diff)
    den = np.sqrt(np.sum(a_diff**2) * np.sum(b_diff**2))
    return num / den

def extract_latent_phases(points_2d):
    """
    Finds the Center of Mass and returns the polar angle for every point.
    points_2d: (N, 2) array from UMAP/Isomap
    """
    # 1. Find Center of Mass (Centroid)
    center_of_mass = np.mean(points_2d, axis=0)
    
    # 2. Re-center data around (0,0)
    centered_data = points_2d - center_of_mass
    
    # 3. Calculate angles in radians [-pi, pi]
    angles = np.arctan2(centered_data[:, 1], centered_data[:, 0])
    
    # 4. Shift to [0, 2*pi]
    return (angles + 2 * np.pi) % (2 * np.pi)

# ==============================================================================
# 2. ANALYSIS PIPELINE
# ==============================================================================

def analyze_manifold_alignment(points_2d, grating_angles_deg):
    """
    Computes the correlation between 2D manifold structure and stimulus angles.
    """
    # 1. Get Latent Phases from the 2D mapping
    latent_phases = extract_latent_phases(points_2d)
    
    # 2. Convert Stimulus Angles to Radians
    # Note: If gratings are 0-180, we map them to 0-2pi to cover the full circle
    stimulus_radians = np.deg2rad(grating_angles_deg * (360.0 / 180.0))
    
    # 3. Compute Correlation
    # We check both directions because the manifold might be mirrored
    corr_normal = circular_correlation(latent_phases, stimulus_radians)
    corr_mirrored = circular_correlation(2 * np.pi - latent_phases, stimulus_radians)
    
    best_corr = max(corr_normal, corr_mirrored)
    
    print(f"--- Topography Correlation Results ---")
    print(f"Circular Correlation: {best_corr:.4f}")
    
    return latent_phases, best_corr

# ==============================================================================
# 3. VISUALIZATION (ICML Style)
# ==============================================================================

def plot_topography_polar(latent_phases, grating_angles_deg, corr_score):
    """
    Plots the latent angles recovered from the manifold.
    """
    plt.figure(figsize=(4, 4))
    ax = plt.subplot(111, projection='polar')
    
    # Use hsv map for circular angles
    norm = plt.Normalize(0, 180)
    colors = plt.cm.hsv(norm(grating_angles_deg % 180))
    
    # We use a constant radius of 1 to visualize the Phase Alignment
    r = np.ones_like(latent_phases)
    
    ax.scatter(latent_phases, r, c=colors, s=20, alpha=0.6, edgecolors='none')
    
    ax.set_title(f"Manifold Angular Alignment\nCirc-Corr: {corr_score:.3f}", fontsize=10)
    ax.set_yticklabels([]) # Hide radius ticks
    
    plt.tight_layout()
    plt.savefig("neural_data_polar.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    # 1. Initialize Data Generator
    # We set up the feature dict for Neuropixel data
    gen = DataGenerator(
        **dict(output_representation= "natural",
               features= [
                   {
                        "type": "neural_npixel",
                        "session_id": SESSION_ID,
                        "window_duration": 0.25,
                        "cache_dir": CACHE_DIR,
                        "normalize_neurons": True,
                        "spatial_frequency_filter": 0.04,
                        "num_neurons": N_NEURONS,
                        "window_latency": 0.08
                    }
                ],
                n_components= N_NEURONS,
                noise_normalized_std=0.0,
                batch_size=BATCH_SIZE)
        )

    print(f"Loading Neural Data from Session {SESSION_ID}...")
    # Sample a large enough batch to see the manifold structure
    neural_batch, metadata = gen.sample_batch_of_data(return_hidden_params=True)
    true_angles = metadata[:, 0]  # Orientation of the gratings

    # 2. Grid Search
    results = []
    best_score = -1.0
    best_config = None
    best_data = None

    print("\nStarting Hyperparameter Sweep...")
    print(f"{'Method':<8} | {'Neighbors':<10} | {'MinDist':<8} | {'Circ-Corr':<10}")
    print("-" * 50)

    for method, n_neighbors in product(METHODS, N_NEIGHBORS_LIST):
        # Handle cases where min_dist is irrelevant for Isomap
        current_min_dists = MIN_DISTS if method == 'umap' else [None]
        
        for m_dist in current_min_dists:
            try:
                # Initialize Mapper
                mapper = NeuralMapper(
                    method=method, 
                    n_neighbors=n_neighbors, 
                    min_dist=m_dist if m_dist else 0.1
                )
                
                # Project to 2D
                points_2d = mapper.fit_transform(neural_batch)
                
                # Analyze Alignment
                _, score = analyze_manifold_alignment(points_2d, true_angles)
                
                results.append({
                    'method': method,
                    'n_neighbors': n_neighbors,
                    'min_dist': m_dist,
                    'score': score
                })
                
                print(f"{method:<8} | {n_neighbors:<10} | {str(m_dist):<8} | {score:.4f}")

                # Track best configuration
                if abs(score) > best_score:
                    best_score = abs(score)
                    best_config = (method, n_neighbors, m_dist)
                    best_data = (points_2d, mapper)

            except Exception as e:
                print(f"Failed configuration {method}/{n_neighbors}: {e}")

    # 3. Final Reporting & Visualization
    print("\n" + "="*50)
    print(f"BEST CONFIGURATION FOUND:")
    print(f"Method: {best_config[0]} | Neighbors: {best_config[1]} | MinDist: {best_config[2]}")
    print(f"Highest Correlation: {best_score:.4f}")
    print("="*50)

    # Use the best model to generate final plots
    best_points, best_mapper = best_data
    
    # Plot 1: Standard Manifold Visualization
    best_mapper.visualize(params=metadata, param_idx=0, title=f"Best Manifold ({best_config[0]})")
    
    # Plot 2: Polar Topography Visualization
    latent_phases, _ = analyze_manifold_alignment(best_points, true_angles)
    plot_topography_polar(latent_phases, true_angles, best_score)

    # 4. Optional: Save summary to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("manifold_sweep_results.csv", index=False)
    print("\nSweep results saved to manifold_sweep_results.csv")