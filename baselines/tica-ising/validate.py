import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import the 1D generator and the Ising TICA model
from synthetic_data_generator import DataGenerator
from tica import TICAModelIsing

def get_all_symmetries_ising(y_batch):
    """
    Generates reflections (flips) and sign symmetries for a batch of 1D open chains.
    (No cyclic translations since the Ising chain has hard boundaries).
    """
    transforms = []
    transform_names = []
    
    y_batch = np.array(y_batch)
    
    for flip in [False, True]:
        y_flip = np.flip(y_batch, axis=1) if flip else y_batch
        for sign in [1, -1]:
            y_final = sign * y_flip
            transforms.append(y_final)
            name = f"FlipLR: {flip}, Sign: {sign}"
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

def evaluate_and_align_reconstruction_ising(x_true, y_raw_output):
    """
    Searches for the spatial/sign transformation that maximizes correlation.
    """
    x_true = np.array(x_true)
    y_raw_output = np.array(y_raw_output)
    
    if len(x_true.shape) == 3 and x_true.shape[-1] == 1:
        x_true = np.squeeze(x_true, axis=-1)
        
    transforms, names = get_all_symmetries_ising(y_raw_output)
    
    best_corr = -1.0
    best_transform = None
    best_name = ""
    
    print(f"Scanning symmetry group ({len(transforms)} permutations) for optimal alignment...")
    for y_trans, name in zip(transforms, names):
        corr = compute_batch_pearson_correlation(x_true, y_trans)
        
        if corr > best_corr:
            best_corr = corr
            best_transform = y_trans
            best_name = name
            
    print(f"\n[✔] Best Match Found!")
    print(f"Alignment: {best_name}")
    print(f"Max Correlation: {best_corr:.4f}")
    
    return best_transform, best_corr, best_name

def visualize_results_ising(x_true, y_raw, y_aligned, save_path=None, num_samples=5):
    """
    Plots Original vs Raw Model Output vs Aligned Model Output using step plots.
    """
    plt.figure(figsize=(15, 3 * num_samples))
    x_axis = np.arange(x_true.shape[1])
    
    for i in range(num_samples):
        # 1. Original Signal (Ground Truth)
        ax1 = plt.subplot(num_samples, 3, i * 3 + 1)
        ax1.step(x_axis, np.squeeze(x_true[i]), where='mid', color='black', linewidth=1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_title("Ground Truth Ising" if i == 0 else "")
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 2. Raw Output
        ax2 = plt.subplot(num_samples, 3, i * 3 + 2)
        ax2.step(x_axis, np.squeeze(y_raw[i]), where='mid', color='blue', linewidth=1.5)
        ax2.set_title("Raw Model Output" if i == 0 else "")
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # 3. Aligned Output
        ax3 = plt.subplot(num_samples, 3, i * 3 + 3)
        ax3.step(x_axis, np.squeeze(y_aligned[i]), where='mid', color='green', linewidth=1.5)
        ax3.set_title("Optimally Aligned Output" if i == 0 else "")
        ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Plot successfully saved to: {save_path}")
    
    plt.close()

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Evaluate 1D TICA representations on Ising.")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to evaluate")
    args = parser.parse_args()

    # --- Configuration & Paths ---
    BATCH_SIZE = 250
    N_COMPONENTS = 33 # 33 for Ising
    INPUT_DIM = N_COMPONENTS
    
    # Define dynamic paths
    weights_path = f"experiments/{args.exp_name}/epochs/tica_weights_epoch_{args.epoch}.npy"
    monitor_dir = f"experiments/{args.exp_name}/monitoring/ep{args.epoch}"
    
    # Create the monitoring directory if it doesn't exist
    os.makedirs(monitor_dir, exist_ok=True)
    
    # 1. Initialize Ising Generator
    print("Initializing Data Generator...")
    generator = DataGenerator(
        batch_size=BATCH_SIZE,
        features=[{"type": "ising", "beta_min": 1.0, "beta_max": 5.0, "n_gibbs_steps": 10}],
        n_components=N_COMPONENTS,
        is_circulant=False, # Hard boundaries for Ising
        output_representation="permuted" # Or "linear", matching how you trained
    )

    # 2. Sample Data
    print("Generating batch of data...")
    x_transformed, x_true = generator.sample_batch_of_data(return_hidden_signal=True)
    x_model_input = tf.reshape(x_transformed, [BATCH_SIZE, INPUT_DIM])

    # -------------------------------------------------------------------------
    # IMPORTANT: ZCA Whitening
    # You MUST apply the exact same pre-computed ZCA matrix here.
    # -------------------------------------------------------------------------
    try:
        global_mean = np.load(f"experiments/{args.exp_name}/zca_mean.npy")
        zca_matrix = np.load(f"experiments/{args.exp_name}/zca_matrix.npy")
        x_model_input = tf.matmul((x_model_input - global_mean), zca_matrix)
        print("Applied saved ZCA whitening matrix.")
    except FileNotFoundError:
        print("[!] WARNING: ZCA matrices not found! Ensure model was not trained with ZCA, or provide the files.")

    # 3. Instantiate and Load the TICA Model
    print(f"Loading TICA Model weights from epoch {args.epoch}...")
    model = TICAModelIsing(
        input_dim=INPUT_DIM,
        n_components=N_COMPONENTS,
        pool_size=3,
        use_orthogonal=True,
        is_circulant=False
    )
    
    # Initialize variables
    dummy_input = tf.zeros((1, INPUT_DIM))
    _ = model(dummy_input)
    
    # Load weights safely
    if os.path.exists(weights_path):
        learned_weights = np.load(weights_path)
        model.W.assign(learned_weights)
        print("Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"CRITICAL: Weights file '{weights_path}' not found!")

    # 4. Get Model Predictions
    print("Running inference...")
    y_pred = model(x_model_input) 

    # 5. Evaluate and Align
    y_aligned, max_corr, best_transform_name = evaluate_and_align_reconstruction_ising(
        x_true=x_true, 
        y_raw_output=y_pred
    )

    # 6. Save Metrics to Text File
    metrics_file = os.path.join(monitor_dir, "alignment_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Experiment: {args.exp_name}\n")
        f.write(f"Epoch: {args.epoch}\n")
        f.write(f"Best Alignment: {best_transform_name}\n")
        f.write(f"Max Pearson Correlation: {max_corr:.6f}\n")
    print(f"Metrics saved to: {metrics_file}")

    # 7. Visualize and Save Plot
    plot_path = os.path.join(monitor_dir, "reconstruction_comparison.png")
    visualize_results_ising(
        x_true=np.array(x_true), 
        y_raw=y_pred.numpy(), 
        y_aligned=y_aligned, 
        save_path=plot_path,
        num_samples=5
    )