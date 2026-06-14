import torch
import numpy as np
import argparse
import os
import sys
import itertools
import json
import subprocess
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import circulant
from matplotlib.ticker import StrMethodFormatter
from torch.utils.data import DataLoader, TensorDataset

# Internal Project Modules
from gan import LieGenerator, LieDiscriminator
from synthetic_data_generator import DataGenerator
from train import train_lie_gan, train_lie_gan_incremental

# ==============================================================================
# 0. Spectral Evaluation & Visualization Logic (Integrated)
# ==============================================================================

class SpectralEvaluator:
    def __init__(self, n_pixels):
        self.n = n_pixels
        # Using numpy pi for precision in phase calculation
        self.dft_matrix = self._build_dft_matrix()

    def _build_dft_matrix(self):
        """Constructs a normalized N x N DFT matrix."""
        indices = torch.arange(self.n).view(-1, 1)
        phases = -2j * np.pi * indices * torch.arange(self.n) / self.n
        return torch.exp(phases) / np.sqrt(self.n)

    def get_truncated_basis(self, bandlimit_norm):
        k_cutoff = int(np.round(bandlimit_norm * (self.n // 2)))
        k_cutoff = max(1, k_cutoff)
        return self.dft_matrix[:k_cutoff, :]
    
    def filter_matrix(self, matrix, bandlimit_norm):
        """Projects a matrix onto the band-limited subspace and reconstructs it."""
        F_trunc = self.get_truncated_basis(bandlimit_norm) 
        F_H = F_trunc.t().conj()                           
        
        mat_c = matrix.to(torch.complex64)
        core = F_trunc @ mat_c @ F_H
        recon = F_H @ core @ F_trunc
        return recon.real

    def compute_projected_similarity(self, L_learned, L_ideal, bandlimit_norm):
        F_tilde = self.get_truncated_basis(bandlimit_norm)
        L_l_c = L_learned.to(torch.complex64)
        L_i_c = L_ideal.to(torch.complex64)
        
        proj_l = F_tilde @ L_l_c @ F_tilde.t().conj()
        proj_i = F_tilde @ L_i_c @ F_tilde.t().conj()

        v_l = torch.view_as_real(proj_l.flatten()).flatten()
        v_i = torch.view_as_real(proj_i.flatten()).flatten()
        return torch.nn.functional.cosine_similarity(v_l, v_i, dim=0).item()

def get_ideal_translation_generator(n):
    """Constructs the ideal spectral derivative (d/dx) matrix."""
    k = np.fft.fftfreq(n) * n
    spectral_ramp = 1j * 2 * np.pi * k / n
    first_row = np.fft.ifft(spectral_ramp).real
    return torch.from_numpy(circulant(first_row).T).float()

def load_generator_from_checkpoint(path, device='cpu'):
    """Safely extracts the Lie generator tensor."""
    if not os.path.exists(path):
        print(f"⚠️ Warning: Checkpoint not found at {path}")
        return None
        
    checkpoint = torch.load(path, map_location=device)
    # Handle different saving conventions
    if isinstance(checkpoint, dict):
        L = checkpoint.get('raw_basis', checkpoint.get('Li', None))
        # Fallback: largest parameter if keys missing
        if L is None:
            # Filter for tensors only
            tensors = {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}
            if tensors:
                L = tensors[max(tensors.keys(), key=lambda k: tensors[k].numel())]
    else:
        # Assume whole model or tensor was saved
        L = checkpoint
        
    if L is None: return None
    return L.squeeze().detach()

def _plot_heatmap(ax, data, title, vlim):
    sns.heatmap(
        data, ax=ax, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
        cbar_kws={"format": StrMethodFormatter("{x:+.2f}"), "fraction": 0.046, "pad": 0.04},
        square=True
    )
    ax.set_title(title, pad=12)
    ax.axis('off')

def plot_matrix_comparison(learned, ideal, save_path, title_suffix=""):
    """Plots comparison and saves to save_path."""
    if torch.is_tensor(learned): learned = learned.cpu().numpy()
    if torch.is_tensor(ideal): ideal = ideal.cpu().numpy()

    # Normalization
    max_ideal = np.max(np.abs(ideal)) + 1e-9
    max_learned = np.max(np.abs(learned)) + 1e-9
    
    norm_ideal = ideal / max_ideal
    norm_learned = learned / max_learned
    error = norm_learned - norm_ideal
    
    # Styling
    sns.set_theme(style="whitegrid", rc={'text.color': 'black', 'axes.labelsize': 14, 'axes.titlesize': 16})
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    vlim = 1.0
    _plot_heatmap(axes[0, 0], norm_ideal, "Ideal (Normalized)", vlim)
    _plot_heatmap(axes[0, 1], norm_learned, "Learned (Normalized)", vlim)
    _plot_heatmap(axes[1, 0], error, "Structure Error", vlim)
    
    sns.histplot(error.flatten(), ax=axes[1, 1], kde=False, color=sns.color_palette("Blues")[4])
    axes[1, 1].set_title("Error Histogram", pad=12) 
    axes[1, 1].set_xlim(-1.0, 1.0)
    
    fig.suptitle(f"Lie Generator Comparison{title_suffix}", fontsize=24)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==============================================================================
# Configuration & Utils
# ==============================================================================

TASK_FINITE_TRANSLATION = 'finite-translation-discovery'
TASK_INFINITE_TRANSLATION = 'infinite-translation-discovery'
TASK_DISCRETE_TRANSLATION = 'discrete-translation-discovery'

def parse_arguments():
    parser = argparse.ArgumentParser(description="LieGAN Grid Search Pipeline")
    parser.add_argument('--task', type=str, required=True, choices=[TASK_FINITE_TRANSLATION, 
                                                                    TASK_INFINITE_TRANSLATION, 
                                                                    TASK_DISCRETE_TRANSLATION])
    
    parser.add_argument('--dataset_size', type=int, default=50000)
    parser.add_argument('--n_component', type=int, default=15)
    parser.add_argument('--n_channel', type=int, default=1)
    parser.add_argument('--grid_search', type=bool, default=True)
    parser.add_argument('--lr_g_min', type=float, default=1e-3)
    parser.add_argument('--lr_g_max', type=float, default=1e-3)
    parser.add_argument('--lr_g_steps', type=int, default=1)
    parser.add_argument('--lr_d_min', type=float, default=2e-5)
    parser.add_argument('--lr_d_max', type=float, default=2e-3)
    parser.add_argument('--lr_d_steps', type=int, default=3)
    parser.add_argument('--lambda_min', type=float, default=0.1)
    parser.add_argument('--lambda_max', type=float, default=10.0)
    parser.add_argument('--lambda_steps', type=int, default=3)
    parser.add_argument('--lr_g', type=float, default=1e-3)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--reg_type', type=str, default='cosine')
    parser.add_argument('--p_norm', type=float, default=2)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--incremental', action='store_true')
    parser.add_argument('--model', type=str, default='lie')
    parser.add_argument('--g_init', type=str, default='random')
    parser.add_argument('--coef_dist', type=str, default='normal')
    parser.add_argument('--sigma_init', type=float, default=1.0)
    parser.add_argument('--uniform_max', type=float, default=1.0)
    parser.add_argument('--normalize_Li', action='store_true')
    parser.add_argument('--x_type', type=str, default='vector')
    parser.add_argument('--y_type', type=str, default='scalar')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='saved_model')
    parser.add_argument('--save_name', type=str, default='default')
    return parser.parse_args()

def get_computation_device(gpu_id=0):
    return torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

def set_reproducibility(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except: return "unknown"

def get_unique_run_path(base_dir, crucial_info_str):
    iterator = 0
    while True:
        folder_name = f"{crucial_info_str}_{iterator}"
        full_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        iterator += 1

def save_experiment_config(args, save_path, overrides=None):
    config = vars(args).copy()
    config['git_commit'] = get_git_revision_hash()
    if overrides: config.update(overrides)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

def create_synthetic_dataset(args):
    # (Same implementation as before)
    if args.task == TASK_FINITE_TRANSLATION:
        use_integer_translation = False
        finite_dimensional_shifts = True
    elif args.task == TASK_INFINITE_TRANSLATION:
        use_integer_translation = False
        finite_dimensional_shifts = False
    elif args.task == TASK_DISCRETE_TRANSLATION:
        use_integer_translation = True
        finite_dimensional_shifts = True
    
    features = [{"amplitude_min": 0.5, "amplitude_max": 1.5,
                 "scale_min":0.5, "scale_max":2.5}]
        
    data_gen = DataGenerator(batch_size=args.batch_size, 
                             n_components=args.n_component, 
                             feature_type="gaussian",
                             features=features,
                             use_integer_translation=use_integer_translation,
                             finite_dimensional_shifts=finite_dimensional_shifts,
                             is_circulant=True)
    
    batch_list = [data_gen.sample_batch_of_data() for _ in range(args.dataset_size // args.batch_size)]
    data_array = np.concatenate(batch_list, axis=0)
    return TensorDataset(torch.FloatTensor(data_array), torch.zeros((data_array.shape[0], 1)))

def create_models(args, device):
    discriminator_input_dim = (args.n_component * args.n_channel) + 1 
    generator = LieGenerator(args.n_component, args.n_channel, args).to(device)
    discriminator = LieDiscriminator(discriminator_input_dim).to(device)
    if args.model == 'lie':
        generator.mu.requires_grad = False
        generator.sigma.requires_grad = False
    elif args.model == 'lie_subgrp':
        generator.Li.requires_grad = False
    return generator, discriminator

def get_hyperparameter_grid(args):
    if not args.grid_search: return [(args.lr_g, args.lr_d, args.lamda)]
    return list(itertools.product(
        np.logspace(np.log10(args.lr_g_min), np.log10(args.lr_g_max), args.lr_g_steps),
        np.logspace(np.log10(args.lr_d_min), np.log10(args.lr_d_max), args.lr_d_steps),
        np.logspace(np.log10(args.lambda_min), np.log10(args.lambda_max), args.lambda_steps)
    ))

# ==============================================================================
# Benchmarking Execution
# ==============================================================================

def run_benchmarks(run_path, best_epoch, final_epoch, n_component, device='cpu'):
    """
    Runs spectral evaluation for the Best and Final epochs.
    """
    benchmark_dir = os.path.join(run_path, "benchmarks")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    epochs_to_eval = {
        'best': best_epoch,
        'final': final_epoch
    }
    
    L_ideal = get_ideal_translation_generator(n_component).to(device)
    evaluator = SpectralEvaluator(n_component)
    
    results = {}
    
    for label, epoch in epochs_to_eval.items():
        # Assume standard naming: generator_{epoch}.pt
        ckpt_path = os.path.join(run_path, f"generator_{epoch}.pt")
        L_learned = load_generator_from_checkpoint(ckpt_path, device)
        
        if L_learned is None:
            print(f"Skipping {label} benchmark: Checkpoint {ckpt_path} missing.")
            continue
            
        # 1. Compute Metric (Spectral Similarity)
        # Use a standard bandlimit of 0.75 for evaluation
        score = evaluator.compute_projected_similarity(L_learned, L_ideal, bandlimit_norm=1.0)
        results[f"{label}_spectral_sim"] = score
        
        # 2. Generate Plots (Projected)
        L_learned_filt = evaluator.filter_matrix(L_learned, bandlimit_norm=1.0)
        L_ideal_filt = evaluator.filter_matrix(L_ideal, bandlimit_norm=1.0)
        
        plot_name = os.path.join(benchmark_dir, f"comparison_{label}_ep{epoch}.png")
        plot_matrix_comparison(L_learned_filt, L_ideal_filt, plot_name, title_suffix=f" ({label.title()} Epoch {epoch})")
        
    # Save quantitative results
    with open(os.path.join(benchmark_dir, "scores.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Benchmarks saved to {benchmark_dir}")

# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_arguments()
    device = get_computation_device(args.gpu)
    set_reproducibility(args.seed)
    
    dataset = create_synthetic_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    param_grid = get_hyperparameter_grid(args)
    base_save_path = os.path.join(args.save_path, args.task)
    
    for i, (curr_lr_g, curr_lr_d, curr_lamda) in enumerate(param_grid):
        print(f"\n🧪 Experiment {i+1}/{len(param_grid)} | G:{curr_lr_g:.1e} D:{curr_lr_d:.1e} L:{curr_lamda:.1f}")
        
        crucial_info = f"lrg_{curr_lr_g:.1e}_lrd_{curr_lr_d:.1e}_lam_{curr_lamda:.1f}"
        run_save_path = get_unique_run_path(base_save_path, crucial_info)
        
        overrides = {'lr_g': curr_lr_g, 'lr_d': curr_lr_d, 'lamda': curr_lamda}
        save_experiment_config(args, run_save_path, overrides)
        
        set_reproducibility(args.seed) 
        generator, discriminator = create_models(args, device)
        trainer = train_lie_gan_incremental if args.incremental else train_lie_gan
        
        # ------------------------------------------------------------------
        # Training Phase
        # ------------------------------------------------------------------
        # Assuming trainer returns history: {'G_losses': [...], 'D_losses': [...]}
        history = trainer(
            generator, discriminator, dataloader,
            num_epochs=args.num_epochs, lr_d=curr_lr_d, lr_g=curr_lr_g,       
            lamda=curr_lamda, reg_type=args.reg_type, p_norm=args.p_norm,
            mu=args.mu, eta=args.eta, device=device, task=args.task,
            save_path=run_save_path + '/', print_every=args.print_every,
        )
        
        # ------------------------------------------------------------------
        # Post-Training Analysis
        # ------------------------------------------------------------------
        if history and 'G_losses' in history:
            # 1. Save Loss History
            loss_file = os.path.join(run_save_path, "loss_history.csv")
            with open(loss_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'G_Loss', 'D_Loss'])
                for ep, (g, d) in enumerate(zip(history['G_losses'], history.get('D_losses', []))):
                    writer.writerow([ep, g, d])
            
            # 2. Identify Epochs
            g_losses = history['G_losses']
            
            # [FIX]: Ignore the first 20% of training (or first 5 epochs) as warmup.
            # This prevents picking Epoch 0 where D is random/weak.
            warmup_period = max(5, int(len(g_losses) * 0.2))
            
            if len(g_losses) > warmup_period:
                # Find min index in the sliced array, then add the offset back
                best_epoch_relative = np.argmin(g_losses[warmup_period:])
                best_epoch = int(best_epoch_relative + warmup_period)
            else:
                # Fallback for very short runs
                best_epoch = int(np.argmin(g_losses))

            final_epoch = len(g_losses) - 1
            
            print(f"📊 Analysis: Best Epoch {best_epoch} (Loss {g_losses[best_epoch]:.4f}), Final Epoch {final_epoch}")
            
            # 3. Run Benchmarks
            run_benchmarks(run_save_path, best_epoch, final_epoch, args.n_component, device)

if __name__ == '__main__':
    main()
    
