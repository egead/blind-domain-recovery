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
    if isinstance(checkpoint, dict):
        L = checkpoint.get('raw_basis', checkpoint.get('Li', None))
        if L is None:
            tensors = {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}
            if tensors:
                L = tensors[max(tensors.keys(), key=lambda k: tensors[k].numel())]
    else:
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

    max_ideal = np.max(np.abs(ideal)) + 1e-9
    max_learned = np.max(np.abs(learned)) + 1e-9
    
    norm_ideal = ideal / max_ideal
    norm_learned = learned / max_learned
    error = norm_learned - norm_ideal
    
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
    parser = argparse.ArgumentParser(description="LieGAN Evaluation Pipeline")
    parser.add_argument('--task', type=str, required=True, choices=[TASK_FINITE_TRANSLATION, 
                                                                    TASK_INFINITE_TRANSLATION, 
                                                                    TASK_DISCRETE_TRANSLATION])
    parser.add_argument('--dataset_size', type=int, default=50000)
    parser.add_argument('--n_component', type=int, default=15)
    parser.add_argument('--n_channel', type=int, default=1)
    
    # Defaults in case no config is found
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

def save_experiment_config(args, save_path, overrides=None):
    config = vars(args).copy()
    config['git_commit'] = get_git_revision_hash()
    if overrides: config.update(overrides)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

def create_synthetic_dataset(args):
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

# ==============================================================================
# Benchmarking Execution
# ==============================================================================

def run_benchmarks(run_path, best_epoch, final_epoch, n_component, device='cpu'):
    benchmark_dir = os.path.join(run_path, "benchmarks")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    epochs_to_eval = {'best': best_epoch, 'final': final_epoch}
    L_ideal = get_ideal_translation_generator(n_component).to(device)
    evaluator = SpectralEvaluator(n_component)
    
    results = {}
    
    for label, epoch in epochs_to_eval.items():
        ckpt_path = os.path.join(run_path, f"generator_{epoch}.pt")
        L_learned = load_generator_from_checkpoint(ckpt_path, device)
        
        if L_learned is None:
            continue
            
        score = evaluator.compute_projected_similarity(L_learned, L_ideal, bandlimit_norm=1.0)
        results[f"{label}_spectral_sim"] = score
        
        L_learned_filt = evaluator.filter_matrix(L_learned, bandlimit_norm=1.0)
        L_ideal_filt = evaluator.filter_matrix(L_ideal, bandlimit_norm=1.0)
        
        plot_name = os.path.join(benchmark_dir, f"comparison_{label}_ep{epoch}.png")
        plot_matrix_comparison(L_learned_filt, L_ideal_filt, plot_name, title_suffix=f" ({label.title()} Epoch {epoch})")
        
    with open(os.path.join(benchmark_dir, "scores.json"), "w") as f:
        json.dump(results, f, indent=4)

# ==============================================================================
# Multi-Seed Evaluation Logic
# ==============================================================================

def find_best_configuration(base_search_dir):
    """Scans existing directories to find the config with the highest spectral similarity."""
    best_score = -1.0
    best_config = None
    best_dir = None

    if not os.path.exists(base_search_dir):
        return None, best_score, None

    for run_dir in os.listdir(base_search_dir):
        full_run_dir = os.path.join(base_search_dir, run_dir)
        
        if not os.path.isdir(full_run_dir) or run_dir == "multi_seed_eval":
            continue

        score_path = os.path.join(full_run_dir, "benchmarks", "scores.json")
        config_path = os.path.join(full_run_dir, "config.json")

        if os.path.exists(score_path) and os.path.exists(config_path):
            try:
                with open(score_path, 'r') as f:
                    scores = json.load(f)
                
                score = scores.get('final_spectral_sim', -1.0)

                if score > best_score:
                    best_score = score
                    with open(config_path, 'r') as f:
                        best_config = json.load(f)
                    best_dir = full_run_dir
            except Exception as e:
                pass
                
    return best_config, best_score, best_dir

def run_single_experiment(args, lr_g, lr_d, lamda, seed, run_save_path, device):
    """Encapsulates a single training and benchmarking run."""
    set_reproducibility(seed)
    
    dataset = create_synthetic_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    generator, discriminator = create_models(args, device)
    trainer = train_lie_gan_incremental if args.incremental else train_lie_gan
    
    history = trainer(
        generator, discriminator, dataloader,
        num_epochs=args.num_epochs, lr_d=lr_d, lr_g=lr_g,       
        lamda=lamda, reg_type=args.reg_type, p_norm=args.p_norm,
        mu=args.mu, eta=args.eta, device=device, task=args.task,
        save_path=run_save_path + '/', print_every=args.print_every,
    )
    
    if history and 'G_losses' in history:
        loss_file = os.path.join(run_save_path, "loss_history.csv")
        with open(loss_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'G_Loss', 'D_Loss'])
            for ep, (g, d) in enumerate(zip(history['G_losses'], history.get('D_losses', []))):
                writer.writerow([ep, g, d])
        
        g_losses = history['G_losses']
        warmup_period = max(5, int(len(g_losses) * 0.2))
        
        if len(g_losses) > warmup_period:
            best_epoch_relative = np.argmin(g_losses[warmup_period:])
            best_epoch = int(best_epoch_relative + warmup_period)
        else:
            best_epoch = int(np.argmin(g_losses))

        final_epoch = len(g_losses) - 1
        
        run_benchmarks(run_save_path, best_epoch, final_epoch, args.n_component, device)

        score_file = os.path.join(run_save_path, "benchmarks", "scores.json")
        if os.path.exists(score_file):
            with open(score_file, 'r') as f:
                scores = json.load(f)
            return scores.get('best_spectral_sim', -1.0)
            
    return -1.0

def main():
    args = parse_arguments()
    device = get_computation_device(args.gpu)
    
    base_save_path = os.path.join(args.save_path, args.task)

    # ==========================================================================
    # Phase 1: Search Existing Runs for Best Configuration
    # ==========================================================================
    print("\n" + "="*50)
    print(" PHASE 1: PARSING EXISTING GRID SEARCH ")
    print("="*50)
    
    print(f"🔍 Searching for the best configuration in {base_save_path}...")
    best_config, best_score, best_dir = find_best_configuration(base_save_path)
    
    if best_config is None:
        print("❌ Could not find any valid grid search results.")
        print("Please ensure the directory contains previous runs with 'benchmarks/scores.json' and 'config.json'.")
        return

    opt_lr_g = best_config.get('lr_g', args.lr_g)
    opt_lr_d = best_config.get('lr_d', args.lr_d)
    opt_lamda = best_config.get('lamda', args.lamda)

    print(f"🏆 BEST CONFIGURATION FOUND")
    print(f"📂 Directory: {best_dir}")
    print(f"📈 Best Spectral Sim: {best_score:.4f}")
    print(f"⚙️ Params -> G: {opt_lr_g:.1e}, D: {opt_lr_d:.1e}, L: {opt_lamda:.1f}")

    # ==========================================================================
    # Phase 2: Multi-Seed Evaluation
    # ==========================================================================
    print("\n" + "="*50)
    print(" PHASE 2: MULTI-SEED STATISTICAL EVALUATION ")
    print("="*50)
    
    evaluation_seeds = [42, 123, 456, 789, 999]
    seed_eval_path = os.path.join(base_save_path, "multi_seed_eval")
    os.makedirs(seed_eval_path, exist_ok=True)
    
    seed_scores = []
    
    for i, seed in enumerate(evaluation_seeds):
        print(f"\n🌱 Seed Experiment {i+1}/{len(evaluation_seeds)} | Seed: {seed}")
        
        run_save_path = os.path.join(seed_eval_path, f"seed_{seed}")
        os.makedirs(run_save_path, exist_ok=True)
        
        overrides = {'lr_g': opt_lr_g, 'lr_d': opt_lr_d, 'lamda': opt_lamda, 'seed': seed}
        save_experiment_config(args, run_save_path, overrides)
        
        score = run_single_experiment(args, opt_lr_g, opt_lr_d, opt_lamda, seed, run_save_path, device)
        seed_scores.append(score)
        
        print(f"End of Seed {seed} | Best Spectral Sim: {score:.4f}")

    # ==========================================================================
    # Phase 3: Aggregation and Reporting
    # ==========================================================================
    final_results = {
        "best_hyperparameters": {"lr_g": opt_lr_g, "lr_d": opt_lr_d, "lamda": opt_lamda},
        "source_grid_search_dir": best_dir,
        "seeds": evaluation_seeds,
        "scores": seed_scores,
        "mean_score": float(np.mean(seed_scores)),
        "std_score": float(np.std(seed_scores))
    }
    
    with open(os.path.join(seed_eval_path, "aggregated_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)
        
    print("\n" + "="*50)
    print("✅ MULTI-SEED EVALUATION COMPLETE")
    print(f"Mean Spectral Sim: {final_results['mean_score']:.4f} ± {final_results['std_score']:.4f}")
    print(f"Results saved to: {os.path.join(seed_eval_path, 'aggregated_results.json')}")
    print("="*50)

if __name__ == '__main__':
    main()