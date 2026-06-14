import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

class NeuralMapper:
    """
    Maps (Batch, Neurons) spike-rate data to a 2D plane.
    Designed to work directly with the output of DataGenerator.
    """
    def __init__(self, method='umap', n_neighbors=15, min_dist=0.1):
        self.method = method.lower()
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.embedding = None

    def fit_transform(self, data):
        """
        data: np.ndarray of shape (Batch, Neurons)
        Returns: np.ndarray of shape (Batch, 2)
        """
        # 1. Z-score normalization across the batch
        # Ensures neurons with high firing rates don't bias the distance calculation
        scaled_data = StandardScaler().fit_transform(data)

        # 2. Dimensionality Reduction
        if self.method == 'umap':
            reducer = umap.UMAP(
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                n_components=2,
                metric='cosine' # Robust for rate-based population vectors
            )
        elif self.method == 'isomap':
            reducer = Isomap(
                n_neighbors=self.n_neighbors,
                n_components=2,
                metric='cosine'
            )
        else:
            raise ValueError("Choose 'umap' or 'isomap'")

        self.embedding = reducer.fit_transform(scaled_data)
        return self.embedding

    def visualize(self, params=None, param_idx=0, title="Neural Manifold"):
        """
        params: The metadata/hidden_params from DataGenerator (Batch, Param_Dim)
        param_idx: 0 for Orientation, 1 for Spatial Frequency
        """
        if self.embedding is None:
            raise RuntimeError("Run fit_transform first.")

        plt.figure(figsize=(10, 7))
        
        # Determine color array
        color_values = params[:, param_idx] if params is not None else None
        label = "Orientation" if param_idx == 0 else "Spatial Freq"
        
        scatter = plt.scatter(
            self.embedding[:, 0], 
            self.embedding[:, 1], 
            c=color_values, 
            cmap='hsv' if param_idx == 0 else 'viridis', 
            s=20, 
            alpha=0.8,
            edgecolors='none'
        )

        if color_values is not None:
            plt.colorbar(scatter, label=label)

        plt.title(f"{title} (Colored by {label})")
        plt.xlabel("Manifold 1")
        plt.ylabel("Manifold 2")
        plt.savefig("mapped-neural-signals.png")