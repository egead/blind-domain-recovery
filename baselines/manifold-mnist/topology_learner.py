import numpy as np
import umap
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from typing import Tuple

@dataclass
class TopologyConfig:
    """Hyperparameters for the 2D Topology Recovery algorithm."""
    method: str = 'isomap'
    variance_threshold: float = 0.15
    n_neighbors: int = 4
    grid_height: int = 15
    grid_width: int = 15
    smoothing_coefficient: float = 3.0
    epsilon: float = 1e-8


class TopologyLearner:
    """Recovers the 2D grid structure of shuffled image pixels using manifold learning."""
    
    def __init__(self, config: TopologyConfig):
        self.config = config

    def recover_topology(self, pixel_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes the full pipeline to recover the 2D topology from unstructured data.
        """
        total_pixels = pixel_data.shape[1]
        
        valid_indices, filtered_data = self._filter_low_variance_pixels(pixel_data)
        distance_matrix = self._compute_pseudo_distances(filtered_data)
        embeddings = self._compute_2d_embeddings(distance_matrix)
        aligned_embeddings = self._align_to_grid(embeddings)
        
        convolution_weights = self._compute_convolution_weights(
            aligned_embeddings, valid_indices, total_pixels
        )
        
        return aligned_embeddings, convolution_weights, valid_indices

    def _filter_low_variance_pixels(self, pixel_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        standard_deviations = np.std(pixel_data, axis=0)
        max_std = np.max(standard_deviations)
        threshold = self.config.variance_threshold * max_std
        
        valid_indices = np.where(standard_deviations >= threshold)[0]
        filtered_data = pixel_data[:, valid_indices]
        
        return valid_indices, filtered_data

    def _compute_pseudo_distances(self, data: np.ndarray) -> np.ndarray:
        correlation_matrix = np.corrcoef(data, rowvar=False)
        
        # Prevent log(0) and float inaccuracies
        abs_correlation = np.clip(np.abs(correlation_matrix), self.config.epsilon, 1.0)
        pseudo_distances = -np.log(abs_correlation)
        
        # Ensure strict mathematical symmetry and zero diagonal
        np.fill_diagonal(pseudo_distances, 0.0)
        symmetric_distances = (pseudo_distances + pseudo_distances.T) / 2.0
        
        return symmetric_distances

    def _compute_2d_embeddings(self, distance_matrix: np.ndarray) -> np.ndarray:
        method = self.config.method.lower()
        
        if method == 'isomap':
            embedder = Isomap(
                n_neighbors=self.config.n_neighbors, 
                n_components=2, 
                metric='precomputed'
            )
        elif method == 'umap':
            embedder = umap.UMAP(
                n_neighbors=self.config.n_neighbors, 
                n_components=2, 
                metric='precomputed', 
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported manifold learning method: {method}")
            
        return embedder.fit_transform(distance_matrix)

    def _align_to_grid(self, embeddings: np.ndarray) -> np.ndarray:
        # Rotate coordinates to align with primary axes
        pca = PCA(n_components=2)
        rotated_embeddings = pca.fit_transform(embeddings)
        
        # Scale to match the physical grid dimensions (1-based indexing for math alignment)
        scaler_height = MinMaxScaler(feature_range=(1, self.config.grid_height))
        scaler_width = MinMaxScaler(feature_range=(1, self.config.grid_width))
        
        aligned_embeddings = np.zeros_like(rotated_embeddings)
        aligned_embeddings[:, 0] = scaler_height.fit_transform(rotated_embeddings[:, 0].reshape(-1, 1)).flatten()
        aligned_embeddings[:, 1] = scaler_width.fit_transform(rotated_embeddings[:, 1].reshape(-1, 1)).flatten()
        
        return aligned_embeddings

    def _compute_convolution_weights(
        self, 
        aligned_embeddings: np.ndarray, 
        valid_indices: np.ndarray, 
        total_pixels: int
    ) -> np.ndarray:
        
        weights = np.zeros((self.config.grid_height, self.config.grid_width, total_pixels))
        
        for i in range(1, self.config.grid_height + 1):
            for j in range(1, self.config.grid_width + 1):
                grid_point = np.array([i, j])
                neighbors = self._find_nearest_neighbors(aligned_embeddings, grid_point)
                
                self._assign_rbf_weights_to_grid(
                    weights, i, j, grid_point, aligned_embeddings, neighbors, valid_indices
                )
                
        return weights

    def _find_nearest_neighbors(self, embeddings: np.ndarray, target_point: np.ndarray) -> np.ndarray:
        l1_distances = np.sum(np.abs(embeddings - target_point), axis=1)
        search_radius = 1.0
        neighbors = np.where(l1_distances < search_radius)[0]
        
        while len(neighbors) == 0:
            search_radius += 1.0
            neighbors = np.where(l1_distances < search_radius)[0]
            
        return neighbors

    def _assign_rbf_weights_to_grid(
        self, 
        weights_matrix: np.ndarray, 
        grid_y: int, 
        grid_x: int, 
        grid_point: np.ndarray,
        embeddings: np.ndarray, 
        neighbors: np.ndarray, 
        valid_indices: np.ndarray
    ):
        l2_distances = np.sqrt(np.sum((embeddings[neighbors] - grid_point)**2, axis=1))
        
        # RBF kernel for smoothing
        unnormalized_weights = np.exp(-self.config.smoothing_coefficient * l2_distances)
        weight_sum = np.sum(unnormalized_weights)
        
        for idx, neighbor_idx in enumerate(neighbors):
            original_pixel_idx = valid_indices[neighbor_idx]
            normalized_weight = unnormalized_weights[idx] / weight_sum
            
            # -1 because Python arrays are 0-indexed, but our grid loop is 1-indexed
            weights_matrix[grid_y - 1, grid_x - 1, original_pixel_idx] = normalized_weight


class ImageReconstructor:
    """Handles the rebuilding of images from shuffled pixels."""
    
    @staticmethod
    def reconstruct(shuffled_pixels: np.ndarray, convolution_weights: np.ndarray) -> np.ndarray:
        """
        Maps a 1D array of shuffled pixels back onto a 2D grid using learned weights.
        """
        return np.tensordot(convolution_weights, shuffled_pixels, axes=([2], [0]))