from numpy import pi
import tensorflow as tf
from tensorflow import keras
from tensorflow.linalg import eigh
from core.regularizations import convert_to_regularization_format
from core.lifting_layer import LiftingLayer
from core.symmetry_preserving_bn import SymmetryPreservingBatchNorm
from core.uniformity_estimator import UniformityEstimator
from core.probability_estimator import ProbabilityEstimator


@tf.keras.utils.register_keras_serializable()
class Model(keras.Model):
    def __init__(
        self,
        name="gsl_model",
        use_linear_map=False,
        lifting_space_dims=None,
        lifting_layer_params=None,
        uniformity_estimator_params=None,
        probability_estimator_params=None,
        compute_only_uniformity=None,
        boundary_sizes=None,
        use_svd_pv_computation=False,
        normalization_noise=[None],
        linear_map_recondition_period=10,
        num_lenses=1,
        dilations=[1],
        intrinsic_dimensionality=None,
        covariance_shift=1e-5,
        tcr_noise_var=1e-4,
        *args,
        **kwargs
    ):
        super(Model, self).__init__(name=name, *args, **kwargs)
        self.lifting_layer_params = lifting_layer_params
        self.uniformity_estimator_params = uniformity_estimator_params
        self.probability_estimator_params = probability_estimator_params
        self.compute_only_uniformity = compute_only_uniformity
        self.lifting_space_dims = lifting_space_dims
        self.use_linear_map = use_linear_map
        self.use_svd_pv_computation = use_svd_pv_computation
        self.normalization_noise = normalization_noise
        self.linear_map_recondition_period = linear_map_recondition_period
        self.num_lenses = num_lenses
        self.dilations = dilations
        self.intrinsic_dimensionality = intrinsic_dimensionality
        self.covariance_shift = covariance_shift
        self.tcr_noise_var = tcr_noise_var
        self._args = args
        self._kwargs = kwargs

        if boundary_sizes is None:
            self.boundary_sizes = [0] * num_lenses
        else:
            self.boundary_sizes = boundary_sizes

    def get_config(self):
        config = super(Model, self).get_config()
        config.update({
            "use_linear_map": self.use_linear_map,
            "lifting_space_dims": self.lifting_space_dims,
            "normalization_noise": self.normalization_noise,
            "lifting_layer_params": self.lifting_layer_params,
            "uniformity_estimator_params": self.uniformity_estimator_params,
            "probability_estimator_params": self.probability_estimator_params,
            "compute_only_uniformity": self.compute_only_uniformity,
            "boundary_sizes": self.boundary_sizes,
            "use_svd_pv_computation": self.use_svd_pv_computation,
            "linear_map_recondition_period": self.linear_map_recondition_period,
            "num_lenses": self.num_lenses,
            "dilations": self.dilations,
            "intrinsic_dimensionality": self.intrinsic_dimensionality,
            "covariance_shift": self.covariance_shift,
            "tcr_noise_var": self.tcr_noise_var,
        })
        return config

    def build(self, input_shape=None):
        self._input_shape = input_shape
        self._create_step_counter()
        self._create_input_bn_layer()
        self._create_lifting_layer()
        self._create_uniformity_estimators()

        self._create_probability_estimators()

        if self.use_linear_map:
            self._create_linear_map()

    def call(self, x, lr_scaled_normalized_training_time=None, training=False, analyze=False):
        if lr_scaled_normalized_training_time is None:
            normalized_rank = tf.constant(0.0, dtype=tf.float32)
        else:
            normalized_rank = tf.cast(lr_scaled_normalized_training_time, tf.float32)

        x = tf.reshape(x, [tf.shape(x)[0], -1])
        x = self._bn(x, training)

        if self.use_linear_map and training:
            if self.step_counter % self.linear_map_recondition_period == 0:
                self._condition_linear_map(normalized_rank)

        if self.use_linear_map:
            x = x @ self._linear_map
        elif self.lifting_space_dims is not None:
            padding_size = self.lifting_space_dims - tf.cast(self.n_input_dims, tf.int32)
            x = self._pad_tensor(x, padding_size)

        outputs = self.lifting_layer(x, training=training, analyze=analyze)

        if analyze:
            y, L = outputs[0], outputs[1]
            self._add_losses(y, normalized_rank, training=False)
            return y, L

        y = outputs

        if training:
            self._increase_step_counter()
            self._add_losses(y, normalized_rank, training=True)

        return y

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _add_losses(self, y, normalized_rank, training):
        for l in range(self.num_lenses):
            yd = self._dilate(y, dilation=self.dilations[l])
            yd = self._demean(yd)
            yd = self._normalize(yd, self.normalization_noise[l])

            self.add_loss(self._compute_uniformity_maximization_regularization(l, yd, training))

            if self._tc_loss_enabled(l):
                self.add_loss(self._compute_joint_entropy_maximization_regularization(yd, normalized_rank))
                self.add_loss(self._compute_total_correlation_minimization_regularization(l, yd, normalized_rank, training))

    def _tc_loss_enabled(self, l):
        if self.compute_only_uniformity is not None:
            return not self.compute_only_uniformity[l]
        return True

    def _compute_uniformity_maximization_regularization(self, l, y, training=True):
        y_cropped = self._crop_uniformity_input(l, y)
        uniformity_val = self.uniformity_estimators[l](y_cropped, training=training)
        return convert_to_regularization_format("uniformity_maximization", uniformity_val)

    def _crop_uniformity_input(self, l, y):
        boundary = int(self.boundary_sizes[l])
        dilation = int(self.dilations[l])
        crop = boundary // max(dilation, 1)

        if crop == 0:
            return y

        y_shape = tf.shape(y)
        rank = y.shape.rank
        n_spatial = rank - 2

        begin = [0]
        size = [y_shape[0]]

        for axis in range(n_spatial):
            axis_len = y_shape[axis + 1]
            crop_axis = tf.minimum(crop, (axis_len // 2) - 1)
            crop_axis = tf.maximum(crop_axis, 0)
            begin.append(crop_axis)
            size.append(axis_len - 2 * crop_axis)

        begin.append(0)
        size.append(y_shape[-1])

        return tf.slice(y, begin=begin, size=size)

    def _compute_joint_entropy_maximization_regularization(self, y, normalized_rank):
        pvals, _ = self._compute_principal_components(y)
        if self.intrinsic_dimensionality is not None:
            pvals = pvals[0:self.intrinsic_dimensionality]
        n = tf.shape(pvals)[0]
        h = self._gaussian_entropy_1d(pvals)
        w = self._soft_thresholding_window(normalized_rank, n)
        reg = -tf.reduce_sum(h * w) / tf.reduce_sum(w)
        return convert_to_regularization_format("joint_entropy_maximization", reg)

    def _compute_total_correlation_minimization_regularization(self, l, y, normalized_rank, training=True):
        pvals, _ = self._compute_principal_components(y)
        log_p = self.probability_estimators[l](
            tf.reshape(y, [tf.shape(y)[0], -1, 1]), training=training
        )
        h_marginal = -tf.reduce_mean(log_p)
        h_joint = self._compute_low_rank_entropy_per_dim(pvals, normalized_rank)
        return convert_to_regularization_format("total_correlation_minimization", h_marginal - h_joint)

    def _compute_low_rank_entropy_per_dim(self, pc_vars, normalized_rank):
        if self.intrinsic_dimensionality is not None:
            pc_vars = pc_vars[0:self.intrinsic_dimensionality]
        n = tf.shape(pc_vars)[0]
        h = self._gaussian_entropy_1d(pc_vars)
        w = self._soft_thresholding_window(threshold=normalized_rank, dims=n)
        return tf.reduce_sum(h * w) / tf.reduce_sum(w)

    # ------------------------------------------------------------------
    # Shared math utilities
    # ------------------------------------------------------------------

    def _condition_linear_map(self, normalized_rank, eps=1e-7):
        L = self._linear_map
        dims = tf.minimum(tf.shape(L)[0], tf.shape(L)[1])
        S, U, V = tf.linalg.svd(L, full_matrices=False)
        w = self._soft_thresholding_window(normalized_rank, dims=dims)
        S = S / tf.exp(tf.reduce_sum(w * tf.math.log(S + eps)) / tf.reduce_sum(w))
        self._linear_map.assign(U @ tf.linalg.diag(S) @ tf.linalg.adjoint(V))

    def _cross_covariance(self, x):
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        x = x - tf.reduce_mean(x, axis=0, keepdims=True)
        N = tf.cast(tf.shape(x)[0], tf.float32)
        cov = tf.matmul(x, x, transpose_a=True) / N
        return 0.5 * (cov + tf.transpose(cov))

    def _compute_principal_components(self, x):
        cov = self._cross_covariance(x)
        I = tf.eye(tf.shape(cov)[0], dtype=tf.float32)
        cov_shifted = cov + I * self.covariance_shift

        if self.use_svd_pv_computation:
            pvals = tf.linalg.svd(cov_shifted, compute_uv=False)
            pvecs = None
        else:
            pvals, pvecs = eigh(cov_shifted)
            pvals = pvals[::-1]
            pvecs = pvecs[:, ::-1]

        return pvals, pvecs

    def _dilate(self, x, dilation: int):
        for ax in range(self.group_dims):
            x = self._dilate_axis_with_random_pooling(x, dim_idx=ax, dilation=dilation)
        return x

    def _dilate_axis_with_random_pooling(self, x, dim_idx: int, dilation: int):
        if dilation <= 1:
            return x

        x = self._swap_channels_and_dilation_axis(x, dim_idx=dim_idx)
        n_blocks = tf.shape(x)[-1] // dilation
        flat_leading = tf.reduce_prod(tf.shape(x)[:-1])
        x_r = tf.reshape(x, [flat_leading, n_blocks, dilation])

        rnd = tf.random.uniform(
            shape=[flat_leading, n_blocks],
            minval=0,
            maxval=dilation,
            dtype=tf.int32,
        )

        y = tf.gather(x_r, rnd, batch_dims=2, axis=-1)
        y_shape = tf.concat(
            [tf.shape(x)[:-1], tf.convert_to_tensor([n_blocks], dtype=tf.int32)], axis=0
        )
        y = tf.reshape(y, shape=y_shape)
        return self._swap_channels_and_dilation_axis(y, dim_idx=dim_idx)

    def _swap_channels_and_dilation_axis(self, x, dim_idx):
        if self.group_dims == 1:
            perm = [0, 2, 1]
        elif self.group_dims == 2:
            perm = [0, 3, 2, 1] if dim_idx == 0 else [0, 1, 3, 2]
        elif self.group_dims == 3:
            if dim_idx == 0:
                perm = [0, 4, 2, 3, 1]
            elif dim_idx == 1:
                perm = [0, 1, 4, 3, 2]
            else:
                perm = [0, 1, 2, 4, 3]
        else:
            raise ValueError("Group dims higher than 3 is not implemented.")
        return tf.transpose(x, perm)

    def _gaussian_entropy_1d(self, var):
        log_var = tf.math.log(tf.nn.relu(var) + self.tcr_noise_var)
        return 0.5 * (tf.math.log(2 * pi) + log_var + 1.0)

    def _soft_thresholding_window(self, threshold, dims, gain=75, exp_clamp=50):
        s = tf.linspace(0.0, 1.0, dims)
        exponent = gain * (s - threshold)
        exponent = tf.clip_by_value(exponent, clip_value_min=-exp_clamp, clip_value_max=exp_clamp)
        return 1.0 / (tf.exp(exponent) + 1.0)

    def _increase_step_counter(self):
        incremented_value = self.step_counter + 1
        overflow_mask = tf.cast(incremented_value == 0, tf.uint32)
        next_value = (
            overflow_mask * self.step_counter + (1 - overflow_mask) * incremented_value
        )
        self.step_counter.assign(next_value)

    def _pad_tensor(self, x, padding_size):
        padding_size = tf.cast(padding_size, tf.int32)
        left_pad_size = padding_size // 2
        right_pad_size = padding_size - left_pad_size
        paddings = tf.convert_to_tensor([[0, 0], [left_pad_size, right_pad_size]])
        return tf.pad(x, paddings, "CONSTANT")

    def _demean(self, x):
        return x - tf.reduce_mean(x, keepdims=True)

    def _normalize(self, x, noise, eps=1e-7):
        if noise is None:
            return x
        rms_variance = tf.sqrt(tf.reduce_mean(tf.math.reduce_variance(x, axis=0, keepdims=True)))
        rms_variance_sg = tf.stop_gradient(rms_variance)
        compensation = 1.0 + (noise / (rms_variance_sg + eps))
        return compensation * x / (rms_variance + noise)

    # ------------------------------------------------------------------
    # Layer creation
    # ------------------------------------------------------------------

    def _create_linear_map(self):
        self._linear_map = tf.Variable(
            tf.eye(self.n_input_dims, self.lifting_space_dims),
            trainable=True,
            name="linear_map",
            dtype=tf.float32,
        )

    def _create_lifting_layer(self):
        self.lifting_layer = LiftingLayer(**self.lifting_layer_params)

    def _create_uniformity_estimators(self):
        self.uniformity_estimators = []
        axes_widths = self.lifting_layer.axes_widths
        channels = self.lifting_layer.num_lifted_channels

        for l in range(self.num_lenses):
            dilation = int(self.dilations[l])

            ue = UniformityEstimator(
                name=f"uniformity_estimator_{l}",
                **self.uniformity_estimator_params[l],
            )
            boundary = int(self.boundary_sizes[l])
            dilated_axes = []
            for W in axes_widths:
                W_dil = W // max(dilation, 1)
                crop = boundary // max(dilation, 1)
                W_eff = max(int(W_dil - 2 * crop), 1)
                dilated_axes.append(W_eff)
            in_shape = [self.batchsize] + dilated_axes + [channels]

            ue(tf.zeros(in_shape))
            self.uniformity_estimators.append(ue)

    def _create_probability_estimators(self):
        self.probability_estimators = []
        axes_widths = self.lifting_layer.axes_widths
        channels = self.lifting_layer.num_lifted_channels

        for l in range(self.num_lenses):
            if self.compute_only_uniformity is not None and self.compute_only_uniformity[l]:
                self.probability_estimators.append(None)
                continue

            d = self.dilations[l]

            params_l = None if self.probability_estimator_params is None else self.probability_estimator_params[l]
            if params_l is None:
                pe = ProbabilityEstimator(name=f"probability_estimator_{l}")
            else:
                pe = ProbabilityEstimator(
                    name=f"probability_estimator_{l}",
                    **params_l,
                )

            dilated_axes_widths = [width // d for width in axes_widths]
            in_shape = [self.batchsize] + list(dilated_axes_widths) + [channels]
            dims = tf.reduce_prod(tf.convert_to_tensor(in_shape[1:], dtype=tf.int32))
            pe(tf.zeros(shape=[self.batchsize, dims, 1]))
            self.probability_estimators.append(pe)

    def _create_input_bn_layer(self):
        self._bn = SymmetryPreservingBatchNorm()

    def _create_step_counter(self):
        self.step_counter = tf.Variable(
            tf.zeros([], dtype=tf.uint32), trainable=False, name="step_counter"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def group_dims(self):
        return self.lifting_layer.group_dims

    @property
    def n_input_dims(self):
        return tf.reduce_prod(self._input_shape[1:])

    @property
    def batchsize(self):
        return self._input_shape[0]
