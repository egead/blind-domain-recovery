import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import circulant
from matplotlib.ticker import StrMethodFormatter

# ==============================================================================
# Domain Logic: Spectral Evaluation (Unchanged)
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
        """Maps normalized bandlimit [0,1] to the truncated Fourier basis rows."""
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

# ==============================================================================
# Utility: Matrix Generation & Loading
# ==============================================================================

def get_ideal_translation_generator(n):
    k = np.fft.fftfreq(n) * n
    spectral_ramp = 1j * 2 * np.pi * k / n
    first_row = np.fft.ifft(spectral_ramp).real
    return torch.from_numpy(circulant(first_row).T).float()

def load_generator_from_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    L = checkpoint.get('raw_basis', checkpoint.get('Li', None))
    if L is None:
        L = checkpoint[max(checkpoint.keys(), key=lambda k: checkpoint[k].numel())]
    return L.squeeze().detach()

# ==============================================================================
# Visualization Logic (Matched EXACTLY to TensorFlow Pipeline)
# ==============================================================================

def _plot_heatmap(ax, data, title, vlim):
    """
    Helper to maintain consistent style.
    CRITICAL FIX: 'fraction' and 'pad' ensure colorbar matches plot height.
    """
    sns.heatmap(
        data, ax=ax, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
        cbar_kws={
            "format": StrMethodFormatter("{x:+.2f}"),
            "fraction": 0.046, 
            "pad": 0.04
        },
        square=True # Ensures aspect ratio is 1:1
    )
    # Note: Font size is controlled by the global theme 'axes.titlesize'
    ax.set_title(title, pad=12)
    ax.axis('off')

def plot_matrix_comparison(learned, ideal, title_suffix=""):
    """
    Plots a 2x2 comparison with independent normalization to [-1, 1].
    Matches the 'ExperimentPlotter' style exactly.
    """
    # 1. Convert to numpy
    if torch.is_tensor(learned): learned = learned.detach().cpu().numpy()
    if torch.is_tensor(ideal): ideal = ideal.detach().cpu().numpy()

    # 2. Normalization (Independent)
    max_ideal = np.max(np.abs(ideal)) + 1e-9
    max_learned = np.max(np.abs(learned)) + 1e-9
    
    norm_ideal = ideal / max_ideal
    norm_learned = learned / max_learned
    
    # 3. Compute Error
    error = norm_learned - norm_ideal
    
    # 4. Styling Setup (Copied from TF Pipeline)
    sns.set_theme(
        style="whitegrid",
        rc={
            'text.color': 'black',
            'axes.labelsize': 39, 'axes.labelcolor': 'black',
            'axes.titlesize': 48, 'axes.titlecolor': 'black',
            'legend.fontsize': 39, 'xtick.labelsize': 30, 'ytick.labelsize': 30
        }
    )
    
    # Figure Size: 16x12 matches the TF pipeline
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Titles
    t_i = "Ideal (Normalized)"
    t_l = "Learned (Normalized)"
    t_e = "Structure Error"

    # Plot Heatmaps (Fixed range [-1, 1])
    vlim = 1.0
    _plot_heatmap(axes[0, 0], norm_ideal, t_i, vlim)
    _plot_heatmap(axes[0, 1], norm_learned, t_l, vlim)
    _plot_heatmap(axes[1, 0], error, t_e, vlim)
    
    # Plot Histogram
    sns.histplot(error.flatten(), ax=axes[1, 1], kde=False, color=sns.color_palette("Blues")[4])
    # Title padding and limits to match styling
    axes[1, 1].set_title("Error Histogram", pad=12) 
    axes[1, 1].set_xlim(-1.0, 1.0)
    
    # Suptitle size 52 matches TF pipeline
    fig.suptitle(f"Lie Generator Comparison{title_suffix}", fontsize=52)
    
    plt.tight_layout()
    plt.savefig(f"comparison_filtered{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.show()

# ==============================================================================
# Main Execution
# ==============================================================================

def run_evaluation(model_path, bandlimit_norm=0.5):
    # Setup
    L_learned_raw = load_generator_from_checkpoint(model_path)
    n = L_learned_raw.shape[0]
    L_ideal_raw = get_ideal_translation_generator(n)
    
    # Analysis
    evaluator = SpectralEvaluator(n)
    spec_sim = evaluator.compute_projected_similarity(L_learned_raw, L_ideal_raw, bandlimit_norm)
    
    # Filtering for Visualization
    L_learned_filt = evaluator.filter_matrix(L_learned_raw, bandlimit_norm)
    L_ideal_filt = evaluator.filter_matrix(L_ideal_raw, bandlimit_norm)
    
    print(f"Analysis Complete for N={n}")
    print(f"Bandlimit: {bandlimit_norm}")
    print(f"Spectral Similarity: {spec_sim:.4f}")
    
    # Plotting
    suffix = f" (Low-Pass {bandlimit_norm})"
    plot_matrix_comparison(L_learned_filt, L_ideal_filt, title_suffix=suffix)

if __name__ == "__main__":
    # Update path as needed
    PATH = "/Users/onur/Desktop/Code/GroupDenseLayer/baselines/LieGAN/saved_model/default/lrg_1.0e-03_lrd_2.0e-04_lam_1.0/generator_199.pt"
    run_evaluation(PATH, bandlimit_norm=1.0)