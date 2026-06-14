import numpy as np

class DataGenerator:
    def __init__(
        self,
        batch_size,
        features,
        n_components=33,
        noise_normalized_std=0.0,
        output_representation="natural",
        random_linear_tf_min_eval=0.75,
        random_linear_tf_max_eval=1.33,
        seed=0,
    ):
        self.batch_size = batch_size
        self.features = features
        self.n_components = n_components
        self.noise_normalized_std = noise_normalized_std
        self.output_representation = output_representation
        
        self.random_linear_tf_min_eval = random_linear_tf_min_eval
        self.random_linear_tf_max_eval = random_linear_tf_max_eval
        
        self._batch_counter = 0
            
        np.random.seed(seed)
        
    def reset_batch_counter(self):
        self._batch_counter = 0
    
    def sample_batch_of_data(self, return_hidden_signal=False):
        batch = np.zeros((self.batch_size, self.n_components), dtype=np.float32)
        
        for feature in self.features:
            if feature["type"] != "ising":
                raise ValueError("This pruned generator only supports 'ising' features.")
            
            signal = self._generate_ising(feature)
            batch += signal
        
        batch = self._add_noise(batch)
        final_output = self._apply_output_representation(batch)

        self._batch_counter += 1
        
        if return_hidden_signal:
            return final_output, batch
        else:
            return final_output
            
    # ----------------------------------------------------------------
    # Signal Generation Logic
    # ----------------------------------------------------------------

    def _generate_ising(self, feature):
        return self.lot_ising_1d(
            self.batch_size,
            beta_min=feature.get("beta_min", 1.0),
            beta_max=feature.get("beta_max", 5.0),
            n_gibbs_steps=feature.get("n_gibbs_steps", 10)
        )

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
        elif self.output_representation == "linear":
            return np.einsum("bn, nm->bm", x, self.random_linear_matrix)
        elif self.output_representation == "permuted":
            return np.einsum("bnc, nm->bmc", x[..., None], self.permutation_matrix)[..., 0]
        elif self.output_representation == "dst":
            return np.einsum("bn, nm->bm", x, self.dst_matrix)
        return x

    # ----------------------------------------------------------------
    # Transformations
    # ----------------------------------------------------------------

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
    def permutation_matrix(self, seed=0):
        np.random.seed(seed)
        perm = np.random.permutation(np.arange(self.n_components))
        v = np.eye(self.n_components, 1, dtype=np.float32).T 
        rows = [np.roll(v, shift=p, axis=1) for p in perm]
        return np.concatenate(rows, axis=0)

    @property
    def dst_matrix(self):
        n = np.arange(1, self.n_components + 1, dtype=np.float32)[:, None]
        k = np.arange(1, self.n_components + 1, dtype=np.float32)[None, :]
        dst = np.sin(n * k * np.pi / (self.n_components + 1.0))
        return dst / np.sqrt((self.n_components + 1) / 2)