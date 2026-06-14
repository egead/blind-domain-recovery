import numpy as np
from scipy import special
import tensorflow as tf
from core.neural_data_preprocessors import NeuropixelPreprocessor

class DataGenerator:
    def __init__(
        self,
        batch_size,
        features,
        n_components=33,
        feature_type=None,
        noise_normalized_std=0.0,
        output_representation="natural",
        random_linear_tf_min_eval=0.75,
        random_linear_tf_max_eval=1.33,
        has_composition_map=True,
        has_generator_matrix=True,
        finite_dimensional_shifts=False,
        use_integer_translation=False,
        is_circulant=False,
        is_homogeneous=False,
        seed=0,
        p_exist=0.5,
        num_of_lots=5,
        eps=1e-7,
    ):
        self.batch_size = batch_size
        self.features = features
        self.feature_type = feature_type
        self.n_components = n_components
        self.noise_normalized_std = noise_normalized_std
        self.output_representation = output_representation
        
        self.finite_dimensional_shifts = finite_dimensional_shifts
        self.use_integer_translations = use_integer_translation
        self.is_circulant = is_circulant
        self.is_homogeneous = is_homogeneous
        
        # In homogeneous mode, we strictly use 1 lot (no superposition)
        self.num_of_lots = 1 if self.is_homogeneous else num_of_lots
        self.p_exist = p_exist / len(features)
        
        self.has_composition_map = has_composition_map
        self.has_generator_matrix = has_generator_matrix
        self.random_linear_tf_min_eval = random_linear_tf_min_eval
        self.random_linear_tf_max_eval = random_linear_tf_max_eval
        self.eps = eps
        
        self._batch_counter = 0
        self._fixed_params = {} 
        
        self.timestep_values = self._compute_timestep_values()
            
        np.random.seed(seed)
        
    def reset_batch_counter(self):
        self._batch_counter = 0
    
    def sample_batch_of_data(self, return_hidden_signal=False, return_hidden_params=False):
        batch = np.zeros((self.batch_size, self.n_components), dtype=np.float32)
        
        for i, feature in enumerate(self.features):
            feature = self._generate_feature(feature, feature_idx=i)
            
            if isinstance(feature, tuple):
                signal = feature[0]
                hidden = feature[1]
                if len(feature) > 2:
                    params = feature[2]
            else:
                signal = feature
                hidden = feature
                params = None
                      
            batch += signal
        
        batch = self._add_noise(batch)
        final_output = self._apply_output_representation(batch)

        self._batch_counter += 1
        
        rets = [final_output]
        if return_hidden_signal:
            rets.append(hidden)

        if return_hidden_params:
            rets.append(params)
        
        if len(rets) > 1:
            return tuple(rets)
        else:
            return final_output
            
    # ----------------------------------------------------------------
    # Dispatcher & Signal Generation Logic
    # ----------------------------------------------------------------

    def _generate_feature(self, feature, feature_idx):
        ftype = feature["type"]
        
        if ftype is None and self.feature_type == "seismic_waveforms":
            return self._generate_seismic(feature)

        if ftype == "gaussian":
            return self._generate_gaussian(feature, feature_idx)
        elif ftype == "legendre":
            return self._generate_legendre(feature, feature_idx)
        elif ftype == "sinc":
            return self._generate_sinc(feature, feature_idx)
        elif ftype == "custom":
            return self._generate_custom(feature)
        elif ftype == "ising":
            return self._generate_ising(feature)
        elif ftype == "neural_npixel":
            return self._generate_neural_neuropixel(feature)
        else:
            raise ValueError(f"Feature type '{ftype}' is not defined.")
            
    def _generate_neural_neuropixel(self, feature):
        # 1. FIX: Consistent attribute name (e.g., self._cache)
        if not hasattr(self, "_cache"):
            session_id = feature["session_id"]
            window_duration = feature["window_duration"]
            cache_dir = feature["cache_dir"]
            window_latency = feature["window_latency"]
            frequency_filter = feature.get("spatial_frequency_filter", None)
            orientation_filter = feature.get("orientation_filter", None)
            
            processor = NeuropixelPreprocessor(session_id=session_id, 
                                                window_duration=window_duration,
                                                window_latency=window_latency,
                                                cache_dir=cache_dir)
        
            spks, mov, params = processor.generate_dataset(fixed_frequency=frequency_filter,
                                                        fixed_orientation=orientation_filter)
        
            normalize = feature.get("normalize_neurons", False)
            num_neurons = feature.get("num_neurons", np.shape(spks)[1])
            
            neuron_variance = np.var(spks, axis=0)
            sorted_indices = np.argsort(neuron_variance)[::-1]
            spks = spks[:, sorted_indices]
        
            if num_neurons < np.shape(spks)[1]:                         
                spks = spks[:, :num_neurons]
            
            if normalize:
                spks = self._demean_and_normalize(spks, axis=0)
                
            self._cache = {}
                    
            self._cache["spikes"] = spks
            self._cache["mov"] = mov
            self._cache["params"] = params
            self._cache["n_samples"] = self._cache["spikes"].shape[0]

        # rng setup (Assuming this is correct for your use case)
        rng = np.random.default_rng(seed=self._batch_counter)

        N = self._cache["n_samples"]
        
        # 5. FIX: np.arange
        element_idxs = np.arange(N)
        batch_sample_idxs = rng.choice(element_idxs, size=self.batch_size, replace=True)
        
        x = self._cache["spikes"][batch_sample_idxs]
        x_hidden = self._cache["mov"][batch_sample_idxs]
        
        orientations = self._cache["params"]["orientation"].to_numpy()[batch_sample_idxs]
        spatial_freqes = self._cache["params"]["spatial_frequency"].to_numpy()[batch_sample_idxs]
        params = np.stack([orientations, spatial_freqes], axis=-1)
        
        return (x, x_hidden, params)
        
    def _demean_and_normalize(self, x, axis=0):
        z = (x - np.mean(x, axis=axis, keepdims=True)) / (self.eps + np.std(x, axis=axis, keepdims=True))
        return z
    
    def _generate_sinc(self, feature, feature_idx):
        total_samples = self.batch_size * self.num_of_lots
        
        bws = self._get_shape_parameter(feature, "bandwidth", total_samples, feature_idx)
        amplitudes = self._get_shape_parameter(feature, "amplitude", total_samples, feature_idx)
        centers = self._sample_centers(total_samples)
        
        signal = self._compute_sinc_shape(centers, bws, amplitudes)
        return self._reshape_and_apply_lots(signal)
    
    def _generate_gaussian(self, feature, feature_idx):
        total_samples = self.batch_size * self.num_of_lots
        
        sigmas = self._get_shape_parameter(feature, "scale", total_samples, feature_idx)
        amplitudes = self._get_shape_parameter(feature, "amplitude", total_samples, feature_idx)
        centers = self._sample_centers(total_samples)

        signal = self._compute_gaussian_shape(centers, sigmas, amplitudes)
        return self._reshape_and_apply_lots(signal)

    def _generate_legendre(self, feature, feature_idx):
        total_samples = self.batch_size * self.num_of_lots
        
        lengths = self._get_shape_parameter(feature, "scale", total_samples, feature_idx)
        amplitudes = self._get_shape_parameter(feature, "amplitude", total_samples, feature_idx)
        centers = self._sample_centers(total_samples)
        
        l, m = feature.get("l", 3), feature.get("m", 1)

        signal = self._compute_legendre_shape(centers, lengths, amplitudes, l, m)
        return self._reshape_and_apply_lots(signal)

    def _generate_ising(self, feature):
        return self.lot_ising_1d(
            self.batch_size,
            beta_min=feature.get("beta_min", 1.0),
            beta_max=feature.get("beta_max", 5.0),
            n_gibbs_steps=feature.get("n_gibbs_steps", 10)
        )

    def _generate_custom(self, feature):
        if not hasattr(self, "custom_data"):
            self.load_custom_data(feature["data_path"])
        return self.lot_custom_samples(self.batch_size)

    # ----------------------------------------------------------------
    # Parameter & Helper Strategies
    # ----------------------------------------------------------------

    def _get_shape_parameter(self, feature, param_name, num_samples, feature_idx):
        min_val = feature.get(f"{param_name}_min", 0.5)
        max_val = feature.get(f"{param_name}_max", 1.5)

        if not self.is_homogeneous:
            return np.random.uniform(min_val, max_val, size=[num_samples]).astype(np.float32)

        key = f"{feature_idx}_{param_name}"
        if key not in self._fixed_params:
            self._fixed_params[key] = np.random.uniform(min_val, max_val)
        
        return np.full([num_samples], self._fixed_params[key], dtype=np.float32)

    def _sample_centers(self, num_samples):
        if self.use_integer_translations:
            return np.random.randint(-self.n_components, self.n_components, size=[num_samples]).astype(np.float32)
        else:
            return np.random.uniform(-self.n_components, self.n_components, size=[num_samples]).astype(np.float32)

    def _reshape_and_apply_lots(self, raw_signal):
        """
        Reshapes signal and applies existence probability.
        FIX: In homogeneous mode, we bypass the probability check to ensure signal exists.
        """
        signal = np.reshape(raw_signal, [self.batch_size, self.num_of_lots, self.n_components])
        
        # Force existence for homogeneous benchmarks
        if self.is_homogeneous:
            return np.sum(signal, axis=1)
        
        # Heterogeneous mode: Apply existence probability
        rand_vals = np.random.uniform(0.0, 1.0, size=[self.batch_size, self.num_of_lots])
        logits = np.where(rand_vals < self.p_exist, 1.0, 0.0).astype(np.float32)
        
        return np.sum(signal * logits[:, :, np.newaxis], axis=1)

    def _calculate_distance(self, t, centers):
        """Legacy helper using instance state."""
        return self._calculate_distance_explicit(t, centers, self.n_components)

    def _calculate_distance_explicit(self, t, centers, period_length):
        """Stateless distance calculation for use with arbitrary 't' grids."""
        diff = t - centers
        if self.is_circulant:
            period = float(period_length)
            diff = (diff + period / 2.0) % period - (period / 2.0)
        return diff

    # ----------------------------------------------------------------
    # Mathematical Shape Implementations
    # ----------------------------------------------------------------
    def _compute_sinc_shape(self, centers, bandwidths, amplitudes, t=None):
        if t is None: t = self.timestep_values
        
        # Broadcast 't' for vectorized operations if it's 1D
        if t.ndim == 1: t = t[np.newaxis, :]
        
        centers = centers[..., np.newaxis]
        bandwidths = bandwidths[..., np.newaxis]
        amplitudes = amplitudes[..., np.newaxis]
        
        # Determine period length from the grid size
        period = self.n_components if t.shape[1] == self.n_components else t.shape[1]
        
        if self.finite_dimensional_shifts:
            dist = self._calculate_distance_explicit(t, np.zeros_like(centers), period)
        else:
            dist = self._calculate_distance_explicit(t, centers, period)
            
        # Calculate Analytic Sinc: sin(pi * bw * dist) / (pi * bw * dist)
        # np.sinc handles the singularity at dist=0 automatically.
        y = np.sinc(bandwidths * dist)
            
        if self.finite_dimensional_shifts:
            y = self._apply_fourier_shift(y, centers)
            
        return amplitudes * y
        
    def _compute_gaussian_shape(self, centers, sigmas, amplitudes, t=None):
        if t is None: t = self.timestep_values

        # Broadcast 't' for vectorized operations if it's 1D
        if t.ndim == 1: t = t[np.newaxis, :]
        
        centers = centers[..., np.newaxis]
        sigmas = sigmas[..., np.newaxis]
        amplitudes = amplitudes[..., np.newaxis]
        
        # Determine period length from the grid size
        period = self.n_components if t.shape[1] == self.n_components else t.shape[1]
        
        if self.finite_dimensional_shifts:
            dist = self._calculate_distance_explicit(t, np.zeros_like(centers), period)
        else:
            dist = self._calculate_distance_explicit(t, centers, period)
            
        y = np.exp(-np.square(dist / sigmas) / 2.0)
        y = y / (np.sqrt(2 * np.pi) * sigmas)
            
        if self.finite_dimensional_shifts:
            y = self._apply_fourier_shift(y, centers)
                    
        return amplitudes * y

    def _compute_legendre_shape(self, centers, lengths, amplitudes, l, m, t=None):
        if t is None: t = self.timestep_values
        if t.ndim == 1: t = t[np.newaxis, :]

        centers = centers[..., np.newaxis]
        lengths = lengths[..., np.newaxis]
        amplitudes = amplitudes[..., np.newaxis]
        
        period = self.n_components if t.shape[1] == self.n_components else t.shape[1]
        dist = self._calculate_distance_explicit(t, centers, period)
        
        if self.finite_dimensional_shifts:
            dist = self._calculate_distance_explicit(t, np.zeros_like(centers), period)
        else:
            dist = self._calculate_distance_explicit(t, centers, period)
            
        t_scaled = dist / (lengths * 0.5)
        t_clipped = np.clip(t_scaled, -1.0, 1.0)
        z = self._assoc_legendre_reparam_func(t_clipped, l, m)
            
        if self.finite_dimensional_shifts:
            z = self._apply_fourier_shift(z, centers)
            
        return amplitudes * z

    def _apply_fourier_shift(self, signal, pixels_to_shift):
        """
        Applies shift theorem: F(k) -> F(k) * exp(-i 2 pi k delta)
        """
        # Ensure frequencies exist (safeguard if not in init)
        if not hasattr(self, 'frequencies'):
            self.frequencies = np.fft.fftfreq(self.n_components).astype(np.float32)
             
        spectrum = np.fft.fft(signal)
        # Note: frequencies must match the domain of the FFT
        phase_ramp = np.exp(-2j * np.pi * self.frequencies * pixels_to_shift)
        return np.fft.ifft(spectrum * phase_ramp).real
    
    # ----------------------------------------------------------------
    # External Data Loaders & Samplers
    # ----------------------------------------------------------------
        
    def load_custom_data(self, fpath):
        self.custom_data = np.load(fpath)

    def lot_custom_samples(self, num_samples):
        indices = tf.random.uniform([num_samples], 0, len(self.custom_data), dtype=tf.int32)
        return tf.gather(self.custom_data, indices)

    def lot_seismic_waveform_crops(self, deviation_from_center, num_samples):
        crop_size = self.n_components
        n_ts = tf.shape(self._seismic_data)[1]
        
        indices = tf.random.uniform([num_samples], 0, len(self._seismic_data), dtype=tf.int32)
        waveforms = tf.gather(self._seismic_data, indices)
        
        shifts = tf.random.uniform([num_samples], -deviation_from_center, deviation_from_center + 1, dtype=tf.int32)
        start_indices = (n_ts + shifts) - (crop_size // 2)
        start_indices = tf.clip_by_value(start_indices, 0, n_ts - crop_size)
        
        batch_ids = tf.range(num_samples, dtype=tf.int32)
        time_ids = start_indices[:, None] + tf.range(crop_size, dtype=tf.int32)
        gather_ids = tf.stack([tf.repeat(batch_ids, crop_size), tf.reshape(time_ids, [-1])], axis=1)
        
        crops = tf.gather_nd(waveforms, gather_ids)
        return tf.reshape(crops, [num_samples, crop_size, 1])[..., 0]

    def lot_ising_1d(self, num_samples, beta_min, beta_max, n_gibbs_steps):
        T = 3 * self.n_components
        betas = np.random.uniform(beta_min, beta_max, size=[num_samples]).astype(np.float32)
        xs = np.random.choice([-1, 1], size=(num_samples, T)).astype(np.int8)

        for _ in range(n_gibbs_steps):
            for t in range(T):
                left = xs[:, t-1] if t > 0 else 0
                right = xs[:, t+1] if t < T-1 else 0
                probs = 1.0 / (1.0 + np.exp(-2.0 * betas * (left + right)))
                xs[:, t] = np.where(np.random.rand(num_samples) < probs, 1, -1)
                
        return xs[:, self.n_components:-self.n_components].astype(np.float32)

    # ----------------------------------------------------------------
    # Post-Processing
    # ----------------------------------------------------------------

    def _add_noise(self, x):
        if self.noise_normalized_std <= 0:
            return x
        std = np.std(x, axis=(0, 1), keepdims=True)
        noise = np.random.normal(0.0, 1, x.shape) * std * self.noise_normalized_std
        return x + noise.astype(np.float32)

    def _apply_output_representation(self, x):
        if self.output_representation == "natural":
            return x
        elif self.output_representation == "permuted":
            return np.einsum("bnc, nm->bmc", x[..., None], self.permutation_matrix)[..., 0]
        elif self.output_representation == "dst":
            return np.einsum("bn, nm->bm", x, self.dst_matrix)
        elif self.output_representation == "linear":
            return np.einsum("bn, nm->bm", x, self.random_linear_matrix)
        return x

    # ----------------------------------------------------------------
    # Math Util & Properties
    # ----------------------------------------------------------------

    def _assoc_legendre_reparam_func(self, x, l, m):
        return self._assoc_legendre_func(np.sin(0.5 * np.pi * x), l, m)

    def _assoc_legendre_func(self, x, l, m):
        z = np.zeros_like(x)
        for k in range(m, l + 1):
            c = (self._permutation(k, m) * self._combination(l, k) * special.binom((l + k - 1.0) / 2.0, l))
            z += c * np.power(x, k - m)
        return z * np.power((1.0 - np.square(x)), m / 2.0)

    def _combination(self, n, k):
        return special.factorial(n, exact=True) / (
            special.factorial(n - k, exact=True) * special.factorial(k, exact=True))

    def _permutation(self, n, k):
        return special.factorial(n, exact=True) / special.factorial(n - k, exact=True)

    def _compute_timestep_values(self):
        t = np.linspace(-0.5, 0.5, self.n_components, dtype=np.float32)
        return t * (self.n_components - 1)

    # ----------------------------------------------------------------
    # Analysis & Metrics
    # ----------------------------------------------------------------
    @property
    def permutation_matrix(self, seed=0):
        np.random.seed(seed)
        perm = np.random.permutation(np.arange(self.n_components))
        v = np.eye(self.n_components, 1, dtype=np.float32).T 
        rows = [np.roll(v, shift=p, axis=1) for p in perm]
        return np.concatenate(rows, axis=0)

    @property
    def random_linear_matrix(self):
        np.random.seed(0)
        d = self.n_components
        L = np.random.normal(size=(d, d))
        U, S, Vh = np.linalg.svd(L)
        
        a = (self.random_linear_tf_max_eval - self.random_linear_tf_min_eval) / (np.max(S) - np.min(S))
        b = self.random_linear_tf_min_eval - a * np.min(S)
        S = a * S + b
        
        return U @ np.diag(S) @ Vh

    @property
    def dst_matrix(self):
        n = np.arange(1, self.n_components + 1, dtype=np.float32)[:, None]
        k = np.arange(1, self.n_components + 1, dtype=np.float32)[None, :]
        dst = np.sin(n * k * np.pi / (self.n_components + 1.0))
        return dst / np.sqrt((self.n_components + 1) / 2)

    @property
    def n_features(self):
        return len(self.features)

    # --- RESTORED TEST FUNCTION ---
    @staticmethod
    def test_visualize_transformation(n_comp=15, bandwidth_norm=0.5):
        """
        Verifies the spatial waveforms of the band-limited sinc pulses.
        """
        import matplotlib.pyplot as plt
        
        # Setup specific features for the test
        features = [{
            "type": "sinc",
            "amplitude_min": 1.0, "amplitude_max": 1.0, 
            "bandwidth_min": bandwidth_norm, "bandwidth_max": bandwidth_norm
        }]
        
        gen = DataGenerator(
            batch_size=10, 
            features=features, 
            n_components=n_comp,
            is_homogeneous=True,
            is_circulant=True
        )
        
        # Sample batch (contains stochastic shifts by default)
        batch = gen.sample_batch_of_data()
                
        # 3. Plotting
        plt.figure(figsize=(10, 6))
        
        x_axis = np.arange(n_comp)
        
        # Plot samples to visualize how the sinc pulses are distributed
        for i in range(batch.shape[0]):
            plt.plot(x_axis, batch[i], '-o', label=f'Sample {i}', alpha=0.8, linewidth=1.5)
            
        plt.title(f"Sinc Pulse Samples (Spatial Domain)\nN={n_comp}, Normalized Bandwidth={bandwidth_norm}")
        plt.xlabel("Grid Index (Pixel)")
        plt.ylabel("Amplitude")
        plt.axhline(0, color='black', linewidth=0.8, alpha=0.3)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("sinc_generation_test.png")

class DataGenerator2d:
    def __init__(
        self,
        batch_size,
        features,
        feature_type="mnist",
        latent_x_dims=7,
        latent_y_dims=7,
        noise_normalized_std=0.0,
        output_representation="natural",
        p_exist=0.5,
        num_of_lots=5,
        flatten_output=True,
        eps=1e-6,
        mnist_max_random_translation=28,
        seed=0,
        is_phase_object=False,          
        phase_max=np.pi,                
        phase_observation_mode="near_field", 
        add_object_scattering=False,    
        scattering_phase_max=np.pi,     
        fresnel_distance=0.5,           
        **kwargs
    ):
        self.batch_size = batch_size
        self.features = features
        self.feature_type = feature_type
        self.latent_x_dims = latent_x_dims
        self.latent_y_dims = latent_y_dims
        self.noise_normalized_std = noise_normalized_std
        self.output_representation = output_representation
        self.mnist_max_random_translation = mnist_max_random_translation
        self.p_exist = p_exist / len(features)
        self.num_of_lots = num_of_lots
        self.flatten_output = flatten_output
        self.seed = seed
        self.eps = eps
        
        self.is_phase_object = is_phase_object
        self.phase_max = phase_max
        self.phase_observation_mode = phase_observation_mode
        self.add_object_scattering = add_object_scattering
        self.scattering_phase_max = scattering_phase_max
        self.fresnel_distance = fresnel_distance
        
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        
        self.use_bittensor = False 
        self.use_soft_binning = False
        self.num_soft_bins = 8
        self.soft_binning_sigma = 0.1
        
        if self.feature_type == "mnist":
            (x_train_mnist, _), (x_test_mnist, _) = tf.keras.datasets.mnist.load_data()
            self._mnist_data = np.concatenate([x_train_mnist, x_test_mnist], axis=0)
            
        self.batch_counter = 0
    
    def reset_batch_counter(self):
        self.batch_counter = 0

    def sample_batch_of_data(self, return_hidden_signal=False):
        x = np.zeros(
            shape=[self.batch_size, self.latent_x_dims, self.latent_y_dims], 
            dtype=np.float32
        )
        x_original_object = None 
        
        for feature in self.features:
            self.use_bittensor = feature.get("give_bittensor", False) or feature.get("use_bittensor", False)
            self.use_soft_binning = feature.get("give_soft_binning", False) or feature.get("use_soft_binning", False)
            self.num_soft_bins = feature.get("num_soft_bins", 8)
            self.soft_binning_sigma = feature.get("soft_binning_sigma", -1.0)

            if feature["type"] == "mnist":
                total_length = 0
                samples = []
                while True:
                    x_sampled = self._lot_mnist_crops(
                        max_warping_amplitude = feature.get("max_warping_amplitude", 0.),
                        blackout_probability = feature.get("blackout_probability", 0.),
                        blackout_mask_seed = feature.get("blackout_mask_seed", 0),
                        shading_probability = feature.get("shading_probability", 0.),
                        shaded_pixel_relative_intensity = feature.get("shaded_pixel_relative_intensity", 1.),
                        high_pass_filter = feature.get("use_high_pass_filter", False),
                        high_pass_filter_sigma = feature.get("high_pass_filter_sigma", 10),
                        apply_padding = feature.get("apply_padding", True),
                    )
                    x_sampled = self._filter_blank_images(x_sampled)
                    samples.append(x_sampled)
                    total_length = total_length + len(x_sampled)
                    if total_length >= self.batch_size:
                        break
                x = tf.concat(samples, axis=0)
                x = x[0: self.batch_size]
                x_original_object = tf.identity(x)
                break
                
            elif feature["type"] == "particles_3d":
                x, x_original_object = self._lot_3d_particles(
                    expected_particles=feature.get("expected_particles", 5.0), 
                    z_min=feature.get("z_min", 0.0),
                    z_max=feature.get("z_max", 10.0),
                    base_radius=feature.get("base_radius", 0.5),
                    defocus_rate=feature.get("defocus_rate", 0.3),
                    ground_truth_dims=feature.get("ground_truth_dims", [9, 9, 9])
                )
                break
            else:
                raise ValueError('Feature type is not supported in this pruned generator.')
        
        if x_original_object is None:
            x_original_object = tf.identity(x)
            
        x = self._add_noise(x)
        
        if self.use_bittensor:
            x_transformed = self._convert_grayscale_to_bittensor(x)
        elif self.use_soft_binning:
            x_transformed = self._apply_soft_binning(x)
        else:
            x_transformed = tf.expand_dims(x, axis=-1)
            
        if self.output_representation == "natural":
            x_transformed = x_transformed
        elif self.output_representation == "permuted":
            x_transformed = self._apply_permutation_map(x_transformed)
        else:
            raise ValueError('Unsupported representation type. Use "natural" or "permuted".')
                
        if self.flatten_output:
            x_transformed = tf.reshape(x_transformed, shape=[self.batch_size, -1])
                
        self.batch_counter += 1
        
        if return_hidden_signal:
            return x_transformed, x, x_original_object
        else:
            return x_transformed

    def _lot_3d_particles(self, expected_particles=10.0, z_min=0.0, z_max=10.0, base_radius=0.5, defocus_rate=0.3, ground_truth_dims=[9, 9, 9]):
        H = tf.cast(self.latent_x_dims, tf.float32)
        W = tf.cast(self.latent_y_dims, tf.float32)
        
        # 1. Edge Effects Padding
        max_sigma = base_radius + (defocus_rate * z_max)
        pad = 3.0 * max_sigma
        
        original_area = H * W
        padded_area = (H + 2.0 * pad) * (W + 2.0 * pad)
        density_multiplier = padded_area / original_area
        expected_padded = expected_particles * density_multiplier
        
        N_max = int(expected_padded + 5.0 * np.sqrt(expected_padded) + 10)
        
        counts = tf.random.poisson(shape=[self.batch_size], lam=expected_padded)
        counts = tf.cast(tf.clip_by_value(counts, 0, N_max), tf.int32)
        
        mask = tf.sequence_mask(counts, maxlen=N_max, dtype=tf.float32)
        mask = tf.reshape(mask, [self.batch_size, N_max, 1, 1])
        
        pos_x = tf.random.uniform([self.batch_size, N_max, 1, 1], minval=-pad, maxval=W + pad)
        pos_y = tf.random.uniform([self.batch_size, N_max, 1, 1], minval=-pad, maxval=H + pad)
        pos_z = tf.random.uniform([self.batch_size, N_max, 1, 1], minval=z_min, maxval=z_max)
        
        # 2. Render Gaussian PSFs to Sensor Grid
        sigma = base_radius + (defocus_rate * pos_z)
        sigma_sq = tf.square(sigma) + self.eps
        peak_intensity = 1.0 / sigma_sq  
        
        x_grid = tf.cast(tf.range(self.latent_y_dims), tf.float32)
        y_grid = tf.cast(tf.range(self.latent_x_dims), tf.float32)
        X, Y = tf.meshgrid(x_grid, y_grid)
        
        X = tf.reshape(X, [1, 1, self.latent_x_dims, self.latent_y_dims])
        Y = tf.reshape(Y, [1, 1, self.latent_x_dims, self.latent_y_dims])
        
        dist_sq = tf.square(X - pos_x) + tf.square(Y - pos_y)
        particle_intensities = peak_intensity * tf.exp(-dist_sq / (2.0 * sigma_sq))
        
        particle_intensities = particle_intensities * mask
        sensor_image = tf.reduce_sum(particle_intensities, axis=1)
        
        img_min = tf.reduce_min(sensor_image, axis=[1, 2], keepdims=True)
        img_max = tf.reduce_max(sensor_image, axis=[1, 2], keepdims=True) + self.eps
        sensor_image_norm = ((sensor_image - img_min) / (img_max - img_min)) * 255.0
        
        # 3. Create 3D Ground Truth Tensor 
        gt_z_dim, gt_y_dim, gt_x_dim = ground_truth_dims
        
        # Normalize continuous coordinates to [0, 1] relative to the original unpadded FOV
        norm_x = pos_x[:, :, 0, 0] / W
        norm_y = pos_y[:, :, 0, 0] / H
        norm_z = (pos_z[:, :, 0, 0] - z_min) / (z_max - z_min + self.eps)

        # Scale to discrete 3D grid indices
        idx_x = tf.cast(tf.floor(norm_x * gt_x_dim), tf.int32)
        idx_y = tf.cast(tf.floor(norm_y * gt_y_dim), tf.int32)
        idx_z = tf.cast(tf.floor(norm_z * gt_z_dim), tf.int32)

        # Build boolean mask to filter out particles spawned in the padding (outside the sensor FOV)
        valid_mask = (mask[:, :, 0, 0] > 0.5) & \
                     (idx_x >= 0) & (idx_x < gt_x_dim) & \
                     (idx_y >= 0) & (idx_y < gt_y_dim) & \
                     (idx_z >= 0) & (idx_z < gt_z_dim)

        batch_idx = tf.broadcast_to(tf.range(self.batch_size)[:, tf.newaxis], [self.batch_size, N_max])

        # Extract only the valid in-frame coordinates
        valid_b = tf.boolean_mask(batch_idx, valid_mask)
        valid_z = tf.boolean_mask(idx_z, valid_mask)
        valid_y = tf.boolean_mask(idx_y, valid_mask)
        valid_x = tf.boolean_mask(idx_x, valid_mask)

        scatter_indices = tf.stack([valid_b, valid_z, valid_y, valid_x], axis=1)
        scatter_updates = tf.ones_like(valid_b, dtype=tf.float32)

        # Drop into discrete 3D tensor (sums multiple particles falling into the same voxel)
        gt_tensor = tf.scatter_nd(scatter_indices, scatter_updates, shape=[self.batch_size, gt_z_dim, gt_y_dim, gt_x_dim])

        return sensor_image_norm, gt_tensor

    def _apply_soft_binning(self, img):
        img_norm = tf.cast(img, tf.float32) / 255.0
        img_expanded = tf.expand_dims(img_norm, axis=-1)
        
        mu = tf.linspace(0.0, 1.0, self.num_soft_bins)
        mu = tf.cast(mu, tf.float32)
        
        if self.soft_binning_sigma <= 0.0:
            spread_factor = 1.0 
            optimal_sigma = spread_factor * (1.0 / max(1.0, float(self.num_soft_bins - 1)))
            sigma_sq = tf.square(tf.cast(optimal_sigma, tf.float32))
        else:
            sigma_sq = tf.square(tf.cast(self.soft_binning_sigma, tf.float32))
        
        binned = tf.exp(-tf.square(img_expanded - mu) / (2.0 * sigma_sq))
        return binned

    def _apply_phase_object_transform(self, x):
        x = tf.cast(x, tf.float32)
        shape = tf.shape(x)
        H = shape[1]
        W = shape[2]

        x_max = tf.reduce_max(x, axis=[1, 2], keepdims=True) + self.eps
        phi = (x / x_max) * self.phase_max
        E = tf.complex(tf.cos(phi), tf.sin(phi))

        if self.phase_observation_mode == "far_field":
            E_out = tf.signal.fft2d(E)
            I = tf.math.square(tf.math.abs(E_out))
        elif self.phase_observation_mode == "near_field":
            E_f = tf.signal.fft2d(E)
            H_dim = tf.cast(H, tf.float32)
            W_dim = tf.cast(W, tf.float32)
            fx = tf.cast(tf.range(H), tf.float32)
            fx = tf.where(fx >= H_dim/2, fx - H_dim, fx) / H_dim
            fy = tf.cast(tf.range(W), tf.float32)
            fy = tf.where(fy >= W_dim/2, fy - W_dim, fy) / W_dim
            FX, FY = tf.meshgrid(fx, fy, indexing='ij')
            f_squared = FX**2 + FY**2
            phase_shift = -np.pi * self.fresnel_distance * f_squared
            H_tf = tf.complex(tf.cos(phase_shift), tf.sin(phase_shift))
            E_out_f = E_f * tf.cast(H_tf, E_f.dtype)
            E_out = tf.signal.ifft2d(E_out_f)
            I = tf.math.square(tf.math.abs(E_out))
        elif self.phase_observation_mode == "interference":
            E_ref = tf.complex(1.0, 0.0)
            I = tf.math.square(tf.math.abs(E + E_ref))
        elif self.phase_observation_mode == "random_tm":
            E_flat = tf.reshape(E, [self.batch_size, -1])
            n_pixels = H * W
            if not hasattr(self, 'TM') or tf.shape(self.TM)[0] != n_pixels:
                n_pixels_f = tf.cast(n_pixels, tf.float32)
                tm_real = tf.random.normal(shape=[n_pixels, n_pixels], stddev=1.0/tf.math.sqrt(n_pixels_f), seed=self.seed)
                tm_imag = tf.random.normal(shape=[n_pixels, n_pixels], stddev=1.0/tf.math.sqrt(n_pixels_f), seed=self.seed)
                self.TM = tf.complex(tm_real, tm_imag)
            E_out_flat = tf.matmul(E_flat, self.TM)
            E_out = tf.reshape(E_out_flat, [self.batch_size, H, W])
            I = tf.math.square(tf.math.abs(E_out))
        else:
            raise ValueError("Unknown phase_observation_mode.")

        I_min = tf.reduce_min(I, axis=[1, 2], keepdims=True)
        I_max = tf.reduce_max(I, axis=[1, 2], keepdims=True) + self.eps
        I_quantized = ((I - I_min) / (I_max - I_min)) * 255.0

        return tf.cast(I_quantized, tf.float32)
    
    def _lot_mnist_crops(self, apply_padding=True, blackout_probability=0.0, 
                         blackout_mask_seed=0, shading_probability=0.0,
                         shading_mask_seed=0, shaded_pixel_relative_intensity=1.0,
                         high_pass_filter=False, high_pass_filter_sigma=10, **kwargs):
        images = self._sample_batch(self._mnist_data)
        
        if high_pass_filter:
            images = self._apply_high_pass_filter(images, high_pass_filter_sigma)
            
        if apply_padding:
            H_original = tf.shape(images)[1]
            W_original = tf.shape(images)[2]
            target_H = H_original + 2 * self.latent_x_dims
            target_W = W_original + 2 * self.latent_y_dims
            
            images = tf.image.pad_to_bounding_box(
                images, offset_height=self.latent_x_dims, offset_width=self.latent_y_dims, 
                target_height=target_H, target_width=target_W
            )
            
        if self.is_phase_object:
            images_squeezed = tf.squeeze(images, axis=-1)
            images_propagated = self._apply_phase_object_transform(images_squeezed)
            images_to_crop = tf.expand_dims(images_propagated, axis=-1)
        else:
            images_to_crop = images
        
        cropped_images = self._randomly_crop_images(
            images=images_to_crop, H_crop_size=self.latent_x_dims, W_crop_size=self.latent_y_dims, 
            apply_padding=False, only_center_crop=kwargs.get("only_center_crop", False)
        )
        
        cropped_images = tf.reshape(cropped_images, [self.batch_size, self.latent_x_dims, self.latent_y_dims])
        
        if blackout_probability > self.eps:
            cropped_images = self._apply_pixel_blackout(images=cropped_images, blackout_probability=blackout_probability, blackout_seed=blackout_mask_seed)
        
        if shading_probability > self.eps:
            cropped_images = self._apply_pixel_shading(images=cropped_images, shading_probability=shading_probability, shaded_pixel_relative_intensity=shaded_pixel_relative_intensity, shading_seed=shading_mask_seed)
        return cropped_images

    def _apply_high_pass_filter(self, x, sigma=10.0):
        x = tf.cast(x, tf.float32)
        if len(x.shape) == 3: x_input = tf.expand_dims(x, axis=-1)
        else: x_input = x

        channels = tf.shape(x_input)[-1]
        kernel_size = int(4 * sigma) | 1 
        kernel = self._create_gaussian_kernel(sigma, kernel_size)
        kernel = tf.tile(kernel, [1, 1, channels, 1])
        
        blurred = tf.nn.depthwise_conv2d(x_input, kernel, strides=[1, 1, 1, 1], padding="SAME")
        high_pass = x_input - blurred
        
        if len(x.shape) == 3: high_pass = tf.squeeze(high_pass, axis=-1)
        return tf.cast(high_pass + 0.5, tf.uint8)
    
    def _create_gaussian_kernel(self, sigma, kernel_size):
        x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
        y = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
        x_grid, y_grid = tf.meshgrid(x, y)
        kernel = tf.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        kernel = kernel / tf.reduce_sum(kernel)
        return kernel[:, :, tf.newaxis, tf.newaxis]
    
    def _randomly_crop_images(self, images, H_crop_size, W_crop_size, apply_padding=False, only_center_crop=False):
        shape = tf.shape(images)
        batch_size, H, W = shape[0], shape[1], shape[2]
        
        if apply_padding:
            H = H + 2 * H_crop_size
            W = W + 2 * W_crop_size
            images = tf.image.pad_to_bounding_box(images, offset_height=H_crop_size, offset_width=W_crop_size, target_height=H, target_width=W)
            
        height_ratio = tf.cast(H_crop_size, tf.float32) / tf.cast(H, tf.float32)
        width_ratio  = tf.cast(W_crop_size, tf.float32) / tf.cast(W, tf.float32)
        
        max_y_start = 1.0 - height_ratio
        max_x_start = 1.0 - width_ratio

        if only_center_crop:
            y1 = tf.fill([batch_size], max_y_start / 2.0)
            x1 = tf.fill([batch_size], max_x_start / 2.0)
        else:
            y1 = tf.random.uniform([batch_size], minval=0.0, maxval=max_y_start)
            x1 = tf.random.uniform([batch_size], minval=0.0, maxval=max_x_start)

        y2, x2 = y1 + height_ratio, x1 + width_ratio
        boxes = tf.stack([y1, x1, y2, x2], axis=1)
        box_indices = tf.range(batch_size)
        crop_size   = [H_crop_size, W_crop_size]
        
        return tf.image.crop_and_resize(images, boxes, box_indices, crop_size)
    
    def _apply_pixel_blackout(self, images, blackout_probability, blackout_seed):
        H, W = tf.shape(images)[1], tf.shape(images)[2]
        blackout = tf.random.uniform(shape=[H, W], seed=blackout_seed) < blackout_probability
        mask = tf.where(blackout, 0., 1.)
        return images * mask[None, :, :]
    
    def _apply_pixel_shading(self, images, shading_probability, shaded_pixel_relative_intensity, shading_seed):
        H, W = tf.shape(images)[1], tf.shape(images)[2]
        blackout = tf.random.uniform(shape=[H, W], seed=shading_seed) < shading_probability
        mask = tf.where(blackout, 0., 1.)
        return images * (mask[None, :, :] + shaded_pixel_relative_intensity * (1. - mask[None, :, :]))
    
    def _sample_batch(self, data):
        random_indices = tf.random.uniform(shape=[self.batch_size], minval=0, maxval=len(data), dtype=tf.int32)
        samples = tf.gather(data, random_indices)
        return tf.expand_dims(samples, axis=-1)

    def _filter_blank_images(self, x_images):
        image_maximums = tf.reduce_max(x_images, axis=(1, 2))
        mask = image_maximums > self.eps
        return tf.boolean_mask(x_images, mask)

    def _add_noise(self, x_batch):
        if self.noise_normalized_std <= 0: return x_batch
        x_batch_std = tf.math.reduce_std(x_batch, keepdims=True)
        noise = (self.noise_normalized_std * x_batch_std * tf.random.normal(mean=0.0, stddev=1., shape=tf.shape(x_batch), dtype=x_batch.dtype))
        return x_batch + noise

    def _apply_permutation_map(self, x):
        num_c = tf.shape(x)[-1]
        x_reshaped = tf.reshape(x, shape=[self.batch_size, self.latent_x_dims * self.latent_y_dims * num_c])
        x_permuted = np.matmul(x_reshaped, self.permutation_matrix)
        return tf.reshape(x_permuted, shape=[self.batch_size, self.latent_x_dims, self.latent_y_dims, num_c])

    def _convert_grayscale_to_bittensor(self, img):
        img = tf.cast(img, tf.uint8)
        img_expanded = tf.expand_dims(img, axis=-1) 
        shifts = tf.cast(tf.range(7, -1, -1, dtype=tf.int32), tf.uint8)
        masks = tf.bitwise.left_shift(tf.cast(1, tf.uint8), shifts)
        masked = tf.bitwise.bitwise_and(img_expanded, masks)
        return tf.where(masked > 0, 1.0, 0.0)
    
    def _convert_bittensor_to_grayscale(self, bit_tensor):
        powers = tf.range(7, -1, -1, dtype=tf.float32)
        weights = tf.pow(2.0, powers)
        reconstructed = tf.reduce_sum(bit_tensor * weights, axis=-1)
        return tf.cast(reconstructed, tf.uint8)
    
    def from_bittensor_to_greyscale(self, x):
        return self._convert_bittensor_to_grayscale(x)

    def deflatten(self, x):
        return np.reshape(x, newshape=[self.batch_size, self.latent_x_dims, self.latent_y_dims, -1])

    @property
    def permutation_matrix(self):
        rng = np.random.RandomState(self.seed)
        
        if self.use_bittensor:
            out_size = 8 * self.n_dims 
        elif self.use_soft_binning:
            out_size = self.num_soft_bins * self.n_dims
        else:
            out_size = self.n_dims
            
        perm = rng.permutation(np.arange(out_size))
        v = np.concatenate([np.ones([1, 1], dtype=np.float32), np.zeros([1, out_size - 1], dtype=np.float32)], axis=1)
        v_translated = [np.roll(v, axis=1, shift=p) for p in perm]
        return np.concatenate(v_translated, axis=0)

    @property
    def n_dims(self):
        return self.latent_x_dims * self.latent_y_dims
    
    @property
    def channels(self):
        if self.use_bittensor: return 8
        if self.use_soft_binning: return self.num_soft_bins
        return 1
    