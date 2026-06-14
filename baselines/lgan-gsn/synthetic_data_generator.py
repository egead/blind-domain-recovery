import numpy as np
from scipy import special

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
        p_exist=0.5,
        num_of_lots=5,
        seed=0,
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
        
    def sample_batch_of_data(self, return_hidden_signal=False):
        batch_signal = np.zeros((self.batch_size, self.n_components), dtype=np.float32)
        
        for i, feature in enumerate(self.features):
            feature_signal = self._generate_feature_signal(feature, feature_idx=i)
            batch_signal += feature_signal
        
        batch_signal = self._add_noise(batch_signal)
        final_output = self._apply_output_representation(batch_signal)

        self._batch_counter += 1
        
        if return_hidden_signal:
            return final_output, batch_signal
        return final_output

    # ----------------------------------------------------------------
    # Dispatcher & Signal Generation Logic
    # ----------------------------------------------------------------

    def _generate_feature_signal(self, feature, feature_idx):
        ftype = feature.get("type", self.feature_type)
        
        if ftype is "seismic_waveforms":
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
        elif ftype == "seismic_waveforms":
            return self._generate_seismic(feature)
        else:
            raise ValueError(f"Feature type '{ftype}' is not defined.")

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

    def _generate_seismic(self, feature):
        if not hasattr(self, "_seismic_data"):
             pass 
             
        return self.lot_seismic_waveform_crops(
            deviation_from_center=feature["deviation_from_center"],
            num_samples=self.batch_size
        )

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