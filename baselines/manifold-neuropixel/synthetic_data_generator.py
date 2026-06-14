import numpy as np
import pandas as pd
from neural_data_preprocessors import NeuropixelPreprocessor

class DataGenerator:
    def __init__(
        self,
        batch_size,
        features,
        n_components=33,
        noise_normalized_std=0.0,
        output_representation="natural",
        seed=0,
        eps=1e-7,
    ):
        self.batch_size = batch_size
        self.features = features
        self.n_components = n_components
        self.noise_normalized_std = noise_normalized_std
        self.output_representation = output_representation
        self.eps = eps
        
        self._batch_counter = 0
        self._cache = {} # Initialized for Neuropixel data storage
            
        np.random.seed(seed)
        
    def reset_batch_counter(self):
        self._batch_counter = 0
    
    def sample_batch_of_data(self, return_hidden_signal=False, return_hidden_params=False):
        # Neuropixel data is usually loaded into the batch directly rather than additive
        # but we maintain the loop structure for compatibility with the features list.
        batch = np.zeros((self.batch_size, self.n_components), dtype=np.float32)
        
        for i, feature in enumerate(self.features):
            # Only processing neural_npixel types
            if feature["type"] == "neural_npixel":
                signal, hidden, params = self._generate_neural_neuropixel(feature)
                batch += signal
            else:
                continue
        
        batch = self._add_noise(batch)
        final_output = self._apply_output_representation(batch)

        self._batch_counter += 1
        
        rets = [final_output]
        if return_hidden_signal:
            rets.append(hidden)
        
        if return_hidden_params:
            rets.append(params)
        
        return tuple(rets) if len(rets) > 1 else final_output

    def _generate_neural_neuropixel(self, feature):
        if "spikes" not in self._cache:
            session_id = feature["session_id"]
            window_duration = feature["window_duration"]
            cache_dir = feature["cache_dir"]
            window_latency = feature["window_latency"]
            frequency_filter = feature.get("spatial_frequency_filter", None)
            orientation_filter = feature.get("orientation_filter", None)
            
            processor = NeuropixelPreprocessor(
                session_id=session_id, 
                window_duration=window_duration,
                window_latency=window_latency,
                cache_dir=cache_dir
            )
            
            spks, mov, params = processor.generate_dataset(
                fixed_frequency=frequency_filter,
                fixed_orientation=orientation_filter
            )
            
            normalize = feature.get("normalize_neurons", False)
            num_neurons = feature.get("num_neurons", np.shape(spks)[1])
            
            # Rank neurons by variance
            neuron_variance = np.var(spks, axis=0)
            sorted_indices = np.argsort(neuron_variance)[::-1]
            spks = spks[:, sorted_indices]
            
            spikes_subset = spks[:, :num_neurons] if num_neurons < np.shape(spks)[1] else spks
                    
            if normalize:
                spikes_subset = self._demean_and_normalize(spikes_subset, axis=0)
            
            self._cache["spikes"] = spikes_subset
            self._cache["mov"] = mov
            self._cache["params"] = params
            self._cache["n_samples"] = self._cache["spikes"].shape[0]

        rng = np.random.default_rng(seed=self._batch_counter)
        N = self._cache["n_samples"]
        
        batch_sample_idxs = rng.choice(np.arange(N), size=self.batch_size, replace=True)
        
        x = self._cache["spikes"][batch_sample_idxs]
        x_hidden = self._cache["mov"][batch_sample_idxs]
        
        # Extract metadata parameters
        orientations = self._cache["params"]["orientation"].to_numpy()[batch_sample_idxs]
        spatial_freqs = self._cache["params"]["spatial_frequency"].to_numpy()[batch_sample_idxs]
        params_out = np.stack([orientations, spatial_freqs], axis=-1)
        
        return (x, x_hidden, params_out)
    
    def _demean_and_normalize(self, x, axis=0):
        z = (x - np.mean(x, axis=axis, keepdims=True)) / (self.eps + np.std(x, axis=axis, keepdims=True))
        return z

    def _add_noise(self, x):
        if self.noise_normalized_std <= 0:
            return x
        std = np.std(x, axis=(0, 1), keepdims=True)
        noise = np.random.normal(0.0, 1, x.shape) * std * self.noise_normalized_std
        return (x + noise).astype(np.float32)

    def _apply_output_representation(self, x):
        # Keeping only the "natural" pass-through for cleanliness, 
        # unless you specifically use the matrix transformations for Neuropixel data.
        if self.output_representation == "natural":
            return x
        return x

    @property
    def n_features(self):
        return len(self.features)