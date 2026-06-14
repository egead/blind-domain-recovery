import tensorflow as tf
from tensorflow.keras.initializers import Constant
from math import pi
from core.regularizations import convert_to_regularization_format

@tf.keras.utils.register_keras_serializable()
class ProbabilityEstimator(tf.keras.Model):
    def __init__(
        self,
        weight_logits_softmax_gain=3.5,
        max_log_variance_magnitude=2.5,
        kernel_center_initial_min=-2.5,
        kernel_center_initial_max=2.5,
        num_kernels=None,
        eps=1e-7,
        name="probability_estimator",
        *args,
        **kwargs
    ):
        super(ProbabilityEstimator, self).__init__(name=name, **kwargs)
        self._num_kernels = num_kernels
        self._kernel_center_initial_min = kernel_center_initial_min
        self._kernel_center_initial_max = kernel_center_initial_max
        self._weight_logits_softmax_gain = weight_logits_softmax_gain
        self._max_log_variance_magnitude = max_log_variance_magnitude
        self._eps = eps

    def get_config(self):
        config = super(ProbabilityEstimator, self).get_config()
        config.update({
            "num_kernels": self._num_kernels, 
            "kernel_center_initial_min": self._kernel_center_initial_min,
            "kernel_center_initial_max": self._kernel_center_initial_max,
            "weight_logits_softmax_gain": self._weight_logits_softmax_gain,
            "max_log_variance_magnitude": self._max_log_variance_magnitude,
            "eps": self._eps
        })
        return config

    def build(self, input_shape=None):
        self._input_shape = input_shape

        if self._num_kernels is None:
            self._num_kernels = 4

        self._create_kernel_centers()
        self._create_weight_logits()
        self._create_log_variance()

        if self._n_dims > 1:
            self._create_cov_diag_map_parametrization()

    def call(self, x, training=False):
        log_p = self._estimate_probabilities(x)

        if training:
            h = self._compute_entropy(log_p)
            self.add_loss(self._compute_entropy_regularization(h))

        return log_p

    def _compute_entropy_regularization(self, h):
        h_mean = tf.reduce_mean(h)

        # FIX: Correctly maps to the Probability Estimator regularization
        return convert_to_regularization_format(
            "probability_estimator_regularization",
            h_mean,
        )
    
    def _add_regularizations(self, h):
        h_mean = tf.reduce_mean(h)
        
        # Note: If layer_losses isn't initialized elsewhere in your code, 
        # consider replacing this with self.add_loss(h_mean)
        if hasattr(self, 'layer_losses'):
            self.layer_losses.clear()
            self.layer_losses.append(h_mean)

    def _compute_entropy(self, log_p):
        h = -tf.reduce_mean(log_p, axis=0)
        return h

    def _estimate_probabilities(self, x):
        x = self._subtract_kernel_centers(x)
        if self._n_dims > 1:
            x = self._apply_covariance_diagonalizing_transformations(x)

        gaussian_probability_estimations = self._gaussian1d(x)

        log_p = self._mix_gaussian_probability_estimations(gaussian_probability_estimations)
        return log_p

    def _apply_covariance_diagonalizing_transformations(self, x):
        transformations = self._compute_diagonalizing_transformations()
        x_transformed = tf.einsum("bnki, nkij->bnkj", x, transformations)
        return x_transformed

    def _mix_gaussian_probability_estimations(self, gaussian_probability_estimations):
        weights = self._compute_gaussian_kernel_weights()

        lgpe = tf.math.log(gaussian_probability_estimations + self._eps)
        lgpe = tf.reduce_sum(lgpe, axis=-1)
        overflow_preventing_offset = tf.reduce_max(lgpe, axis=-1, keepdims=True)
        p_kernels = tf.exp(lgpe - overflow_preventing_offset)
        p = tf.reduce_sum(p_kernels * weights[tf.newaxis, ...], axis=-1)
        log_p = tf.math.log(p + self._eps) + overflow_preventing_offset[..., 0]

        return log_p

    def _compute_diagonalizing_transformations(self):
        ut = self._vector_to_upper_triangular_matrix(self._cov_diag_map_parametrization)

        antisym_matrix = 0.5 * (tf.linalg.matrix_transpose(ut) - ut)
        q = self._cayley_transformation(antisym_matrix)
        return q

    def _compute_gaussian_kernel_weights(self):
        clipped_weight_logits = tf.clip_by_value(self._weight_logits, 
                                                 -self._weight_logits_softmax_gain, 
                                                 self._weight_logits_softmax_gain)
        
        return tf.nn.softmax(clipped_weight_logits, axis=-1)

    def _compute_variance(self):
        clipped_log_var = tf.clip_by_value(self._log_variance, -self._max_log_variance_magnitude, self._max_log_variance_magnitude)
        var = tf.exp(clipped_log_var)
        return var

    def _subtract_kernel_centers(self, x):
        x = x[:, :, tf.newaxis, :]
        kc = self._kernel_centers[tf.newaxis, ...]
        return x - kc

    def _gaussian1d(self, x):
        var = self._compute_variance()
        var = var[tf.newaxis, ...]

        norm_coeff = 1.0 / tf.sqrt(2.0 * pi * var)
        exponent = -0.5 * x * x / var
        gaussians = norm_coeff * tf.exp(exponent)

        return gaussians

    def _cayley_transformation(self, a):
        id = tf.eye(self._n_dims)
        id = id[tf.newaxis, tf.newaxis, :, :]
        q = tf.matmul(id - a, tf.linalg.inv(id + a))
        return q

    def _vector_to_upper_triangular_matrix(self, v):
        zero_pad = tf.zeros([self._n_timesteps, self._num_kernels, self._n_dims])

        a_flattened = tf.concat([v, v[..., ::-1], zero_pad], axis=2)
        a = tf.reshape(
            a_flattened,
            shape=[self._n_timesteps, self._num_kernels, self._n_dims, self._n_dims],
        )
        idxs = tf.range(0, self._n_dims)
        mask = tf.where((idxs[:, None] < idxs[None, :]), 1.0, 0.0)
        return mask[tf.newaxis, tf.newaxis, :, :] * a

    def _demean(self, x):
        return x - tf.reduce_mean(x, axis=0, keepdims=True)

    def _create_weight_logits(self):
        self._weight_logits = tf.Variable(
            Constant(1.0)(shape=[self._n_timesteps, self._num_kernels]),
            trainable=True,
            name="weight_logits",
        )

    def _create_kernel_centers(self):
        # 1. Generate an evenly spaced 1D grid between min and max
        evenly_spaced = tf.linspace(
            start=tf.cast(self._kernel_center_initial_min, tf.float32), 
            stop=tf.cast(self._kernel_center_initial_max, tf.float32), 
            num=self._num_kernels
        )
        
        # 2. Reshape to [1, num_kernels, 1] for broadcasting
        reshaped_centers = tf.reshape(evenly_spaced, [1, self._num_kernels, 1])
        
        # 3. Broadcast to the required shape: [n_timesteps, num_kernels, n_dims]
        initial_centers = tf.broadcast_to(
            reshaped_centers, 
            shape=[self._n_timesteps, self._num_kernels, self._n_dims]
        )
        
        self._kernel_centers = tf.Variable(
            initial_value=tf.cast(initial_centers, tf.float32),
            trainable=True,
            name="kernel_centers",
        )

    def _create_log_variance(self):
        self._log_variance = tf.Variable(
            Constant(1.0)(shape=[self._n_timesteps, self._num_kernels, self._n_dims]),
            trainable=True,
            name="log_variance",
        )

    def _create_cov_diag_map_parametrization(self):
        d = self._n_dims
        n_elements = d * (d - 1) // 2
        self._cov_diag_map_parametrization = tf.Variable(
            Constant(1.0)(shape=[self._n_timesteps, self._num_kernels, n_elements]),
            trainable=True,
            name="cov_diag_map_parametrization",
        )

    @property
    def _n_dims(self):
        return self._input_shape[2]

    @property
    def _n_timesteps(self):
        return self._input_shape[1]

    @property
    def _batch_size(self):
        return self._input_shape[0]