import numpy as np
import pandas as pd
from synthetic_data_generator import DataGenerator
from neural_mapper import NeuralMapper

# Import the necessary constants and functions from your grid search script
from grid_search import (
    analyze_manifold_alignment, 
    SESSION_ID, 
    BATCH_SIZE, 
    N_NEURONS, 
    CACHE_DIR
)

# =============================================================================
# STATISTICAL VALIDATION CONFIGURATION
# =============================================================================
SEEDS = [42, 104, 2026, 777, 9001]

# Plug in the actual best configurations found from your grid search output
BEST_CONFIGS = {
    'umap': {'n_neighbors': 15, 'min_dist': 0.1},
    'isomap': {'n_neighbors': 30, 'min_dist': None} 
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_validation():
    results = {method: [] for method in BEST_CONFIGS.keys()}
    
    print("=" * 60)
    print("STARTING STATISTICAL VALIDATION")
    print("=" * 60)

    for method, config in BEST_CONFIGS.items():
        print(f"\nEvaluating {method.upper()} with config: {config}")
        print("-" * 60)
        
        for seed in SEEDS:
            # 1. Set global seed for reproducibility in UMAP/Isomap
            np.random.seed(seed)
            
            # 2. Initialize Data Generator
            gen = DataGenerator(
                output_representation="natural",
                features=[{
                    "type": "neural_npixel",
                    "session_id": SESSION_ID,
                    "window_duration": 0.25,
                    "cache_dir": CACHE_DIR,
                    "normalize_neurons": True,
                    "spatial_frequency_filter": 0.04,
                    "num_neurons": N_NEURONS,
                    "window_latency": 0.08
                }],
                n_components=N_NEURONS,
                noise_normalized_std=0.0,
                batch_size=BATCH_SIZE,
                seed=seed
            )
            
            # Override the batch counter to force a new random batch per seed
            gen._batch_counter = seed 
            
            # 3. Sample Data
            neural_batch, metadata = gen.sample_batch_of_data(return_hidden_params=True)
            true_angles = metadata[:, 0]
            
            # 4. Initialize Mapper with the best config
            mapper = NeuralMapper(
                method=method,
                n_neighbors=config['n_neighbors'],
                min_dist=config['min_dist'] if config['min_dist'] else 0.1
            )
            
            try:
                # 5. Project to 2D
                points_2d = mapper.fit_transform(neural_batch)
                
                # 6. Analyze Alignment
                # We suppress the print statements from analyze_manifold_alignment visually
                # by just grabbing the returned score.
                _, score = analyze_manifold_alignment(points_2d, true_angles)
                
                results[method].append(score)
                print(f"Seed {seed:<6} | Circular Correlation: {score:.4f}")
                
            except Exception as e:
                print(f"Seed {seed:<6} | FAILED: {e}")

    # =============================================================================
    # SUMMARY STATISTICS
    # =============================================================================
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY (Across 5 Seeds)")
    print("=" * 60)
    
    summary_data = []
    for method in BEST_CONFIGS.keys():
        scores = results[method]
        
        if not scores:
            print(f"{method.upper():<8} | Failed to compute.")
            continue
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        summary_data.append({
            'Method': method.upper(),
            'Mean Circ-Corr': mean_score,
            'Std Dev': std_score
        })
        
        print(f"{method.upper():<8} | Mean: {mean_score:.4f} ± {std_score:.4f}")

    # Save to CSV for reporting
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("statistical_validation_summary.csv", index=False)
    print("\nSummary saved to statistical_validation_summary.csv")

if __name__ == "__main__":
    run_validation()