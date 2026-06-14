import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Import the 1D generator and the updated 1D TICA model
from synthetic_data_generator import DataGenerator
from tica import TICAModel1D

def get_all_symmetries_1d(y_batch, n_components):
    """
    Generates all cyclic translations, reflections (flips), and sign symmetries 
    for a batch of 1D periodic signals.
    """
    transforms = []
    transform_names = []
    
    y_batch = np.array(y_batch)
    
    # For a 1D sequence, check every possible cyclic shift
    for shift in range(n_components):
        y_shifted = np.roll(y_batch, shift=shift, axis=1)
        for flip in [False, True]:
            y_flip = np.flip(y_shifted, axis=1) if flip else y_shifted
            for sign in [1, -1]:
                y_final = sign * y_flip
                transforms.append(y_final)
                name = f"Shift: {shift}, FlipLR: {flip}, Sign: {sign}"
                transform_names.append(name)
                
    return transforms, transform_names

def compute_batch_pearson_correlation(x_true, y_pred):
    """
    Computes the mean Pearson correlation coefficient across the batch.
    """
    x_flat = np.reshape(x_true, (x_true.shape[0], -1))
    y_flat = np.reshape(y_pred, (y_pred.shape[0], -1))
    
    x_centered = x_flat - np.mean(x_flat, axis=1, keepdims=True)
    y_centered = y_flat - np.mean(y_flat, axis=1, keepdims=True)
    
    cov = np.sum(x_centered * y_centered, axis=1)
    x_std = np.linalg.norm(x_centered, axis=1)
    y_std = np.linalg.norm(y_centered, axis=1)
    
    valid_mask = (x_std > 1e-8) & (y_std > 1e-8)
    corr = np.zeros_like(cov)
    corr[valid_mask] = cov[valid_mask] / (x_std[valid_mask] * y_std[valid_mask])
    
    return np.mean(corr)

def evaluate_and_align_reconstruction_1d(x_true, y_raw_output, n_components):
    """
    Searches for the spatial/sign transformation that maximizes correlation.
    """
    x_true = np.array(x_true)
    y_raw_output = np.array(y_raw_output)
    
    if len(x_true.shape) == 3 and x_true.shape[-1] == 1:
        x_true = np.squeeze(x_true, axis=-1)
        
    transforms, names = get_all_symmetries_1d(y_raw_output, n_components)
    
    best_corr = -1.0
    best_transform = None
    best_name = ""
    
    for y_trans, name in zip(transforms, names):
        corr = compute_batch_pearson_correlation(x_true, y_trans)
        
        if corr > best_corr:
            best_corr = corr
            best_transform = y_trans
            best_name = name
            
    return best_transform, best_corr, best_name

def visualize_results_1d(x_true, y_raw, y_aligned, save_path=None, num_samples=5):
    """
    Plots Original vs Raw Model Output vs Aligned Output for 1D signals.
    """
    plt.figure(figsize=(15, 3 * num_samples))
    x_axis = np.arange(x_true.shape[1])
    
    for i in range(num_samples):
        # 1. Original Signal (Ground Truth)
        ax1 = plt.subplot(num_samples, 3, i * 3 + 1)
        ax1.plot(x_axis, np.squeeze(x_true[i]), color='black', linewidth=1.5)
        ax1.set_title("Ground Truth TGSN" if i == 0 else "")
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 2. Raw Output
        ax2 = plt.subplot(num_samples, 3, i * 3 + 2)
        ax2.plot(x_axis, np.squeeze(y_raw[i]), color='blue', linewidth=1.5)
        ax2.set_title("Raw Model Output" if i == 0 else "")
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # 3. Aligned Output
        ax3 = plt.subplot(num_samples, 3, i * 3 + 3)
        ax3.plot(x_axis, np.squeeze(y_aligned[i]), color='green', linewidth=1.5)
        ax3.set_title("Optimally Aligned Output" if i == 0 else "")
        ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close()

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 1D TICA representations across all experiments.")
    parser.add_argument("--base_folder", type=str, default="experiments", help="Base directory containing all experiments")
    parser.add_argument("--epoch", type=int, default=1500, help="Target epoch number to evaluate")
    args = parser.parse_args()

    all_results = []
    
    if not os.path.exists(args.base_folder):
        raise FileNotFoundError(f"Base folder '{args.base_folder}' does not exist.")

    experiment_folders = [d for d in os.listdir(args.base_folder) if os.path.isdir(os.path.join(args.base_folder, d))]
    
    print(f"Found {len(experiment_folders)} experiments in '{args.base_folder}'. Starting evaluation...\n")

    for exp_name in sorted(experiment_folders):
        exp_dir = os.path.join(args.base_folder, exp_name)
        specs_path = os.path.join(exp_dir, "specs.json")
        
        if not os.path.exists(specs_path):
            print(f"[!] Skipping {exp_name}: 'specs.json' not found.")
            continue
            
        # 1. Load Experiment Specifications (ALIGNED WITH TRAINING DEFAULTS)
        with open(specs_path, "r") as f:
            specs = json.load(f)
            
        BATCH_SIZE = specs.get("batch_size", 250)
        N_COMPONENTS = specs.get("n_components", 63)
        POOL_SIZE = specs.get("pool_size", 5)
        USE_ORTHOGONAL = specs.get("use_orthogonal", True)
        IS_CIRCULANT = specs.get("is_circulant", True)
        LEARNING_RATE = specs.get("learning_rate", 0.01)
        DET_PENALTY = specs.get("det_penalty", 0.1)
        FEATURE_TYPE = specs.get("feature_type", "gaussian")
        OUTPUT_REP = specs.get("output_representation", "natural")

        print(f"--- Evaluating {exp_name} ---")
        
        # 2. Initialize 1D Generator based strictly on specs
        if FEATURE_TYPE == "gaussian":
            features = [{"type": "gaussian", "scale_min": 0.5, "scale_max": 2.5, "amplitude_min": 0.5, "amplitude_max": 1.5}]
        else:
            features = [{"type": "legendre", "scale_min": 6.0, "scale_max": 15.0, "amplitude_min": 0.5, "amplitude_max": 1.5, "l": 3, "m": 1},
                        {"type": "legendre", "scale_min": 6.0, "scale_max": 15.0, "amplitude_min": 0.5, "amplitude_max": 1.5, "l": 2, "m": 1}]

        generator = DataGenerator(
            batch_size=BATCH_SIZE,
            features=features,
            n_components=N_COMPONENTS,
            is_circulant=IS_CIRCULANT,
            output_representation=OUTPUT_REP
        )

        # Generate Evaluation Batch
        x_transformed, x_true = generator.sample_batch_of_data(return_hidden_signal=True)
        
        # The generator returns NumPy arrays, matching the training pipeline
        x_model_input = tf.cast(x_transformed, tf.float32)

        # 3. Apply ZCA Whitening (Exactly as done in training)
        zca_mean_path = os.path.join(exp_dir, "epochs", "zca_mean.npy")
        zca_matrix_path = os.path.join(exp_dir, "epochs", "zca_matrix.npy")
        
        try:
            global_mean = tf.convert_to_tensor(np.load(zca_mean_path), dtype=tf.float32)
            zca_matrix = tf.convert_to_tensor(np.load(zca_matrix_path), dtype=tf.float32)
            
            centered_data = x_model_input - global_mean
            x_model_input = tf.matmul(centered_data, zca_matrix)
        except FileNotFoundError:
            print(f"  [!] Skipping {exp_name}: ZCA matrices not found.")
            continue

        # 4. Instantiate and Load the TICA Model (Aligned with specs)
        model = TICAModel1D(
            input_dim=N_COMPONENTS,
            n_components=N_COMPONENTS,
            pool_size=POOL_SIZE,
            use_orthogonal=USE_ORTHOGONAL,
            is_circulant=IS_CIRCULANT,
            det_penalty=DET_PENALTY
        )
        
        # Build model graph with correct dimensions
        _ = model(tf.zeros((1, N_COMPONENTS)))
        
        weights_path = os.path.join(exp_dir, "epochs", f"tica_weights_epoch_{args.epoch}.npy")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(exp_dir, "tica_weights_final.npy")
            
        if os.path.exists(weights_path):
            learned_weights = np.load(weights_path)
            model.W.assign(learned_weights)
        else:
            print(f"  [!] Skipping {exp_name}: Weights file not found.")
            continue

        # 5. Evaluate and Align
        y_pred = model(x_model_input) 
        y_aligned, max_corr, best_transform_name = evaluate_and_align_reconstruction_1d(
            x_true=x_true, 
            y_raw_output=y_pred,
            n_components=N_COMPONENTS
        )
        
        print(f"  Max Correlation: {max_corr:.4f}")

        # Save Plots & Metrics locally inside the experiment monitoring folder
        monitor_dir = os.path.join(exp_dir, "monitoring", f"ep{args.epoch}_eval")
        os.makedirs(monitor_dir, exist_ok=True)
        
        metrics_file = os.path.join(monitor_dir, "alignment_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Experiment: {exp_name}\nEpoch: {args.epoch}\n")
            f.write(f"Feature: {FEATURE_TYPE}\nBest Alignment: {best_transform_name}\n")
            f.write(f"Max Pearson Correlation: {max_corr:.6f}\n")
            
        plot_path = os.path.join(monitor_dir, "reconstruction_comparison.png")
        visualize_results_1d(np.array(x_true), y_pred.numpy(), y_aligned, save_path=plot_path)

        # Append to results for the summary table
        all_results.append({
            "Experiment": exp_name,
            "LR": LEARNING_RATE,
            "Det Penalty": DET_PENALTY,
            "Pool Size": POOL_SIZE,
            "Orthogonal": USE_ORTHOGONAL,
            "Circulant": IS_CIRCULANT,
            "Feature Type": FEATURE_TYPE,
            "Output Rep": OUTPUT_REP,
            "Correlation": max_corr
        })

    # ==============================================================================
    # 6. Generate Summary Table (Filtering for varying parameters)
    # ==============================================================================
    if not all_results:
        print("\nNo successful evaluations to report.")
        exit()

    df = pd.DataFrame(all_results)
    
    # Identify columns that actually vary across experiments
    static_cols = []
    varying_cols = ["Experiment"]
    
    for col in df.columns:
        if col not in ["Experiment", "Correlation"]:
            if df[col].nunique() > 1:
                varying_cols.append(col)
            else:
                static_cols.append(col)
                
    varying_cols.append("Correlation")
    
    # Filter the DataFrame to only show the varying parameters and the score
    summary_df = df[varying_cols].sort_values(by="Correlation", ascending=False)
    
    print("\n" + "="*80)
    print(" EVALUATION SUMMARY (Sorted by Correlation)")
    print("="*80)
    print(f"Static Parameters across all runs: {', '.join(static_cols) if static_cols else 'None'}")
    print("-" * 80)
    
    # Print a clean Markdown-compatible table using Pandas
    print(summary_df.to_markdown(index=False, floatfmt=".4f"))
    print("="*80)
    
    # Save the summary table to the base directory
    summary_csv_path = os.path.join(args.base_folder, f"correlation_summary_ep{args.epoch}.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary table saved to: {summary_csv_path}")