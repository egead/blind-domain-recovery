import tensorflow as tf
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class SpectralNormConstraint(tf.keras.layers.Wrapper):
    """Performs spectral norm constraint on weights.
    """

    @typechecked
    def __init__(self, layer: tf.keras.layers.Layer, max_spectral_norm, conv=True, power_iterations: int = 1, eps=1e-12, **kwargs):
        super().__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero, got "
                "`power_iterations={}`".format(power_iterations)
            )
        self.max_spectral_norm = max_spectral_norm
        self.power_iterations = power_iterations
        self.conv = conv
        
        self._eps = eps
        self._initialized = False

    def build(self, input_shape):
        """Build `Layer`"""
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(
                "{} object has no attribute 'kernel' nor "
                "'embeddings'".format(type(self.layer).__name__)
            )

        self.w_shape = self.w.shape.as_list()
        
        if self.conv:
            self._num_out_channels = self.w_shape[-1]
            
            self.u = self.add_weight(
                shape=(1, self._num_out_channels),
                initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                trainable=False,
                name="sn_u",
                dtype=self.w.dtype,
            )
        else:
            self._out_dims = self.w_shape[1]
            self.u = self.add_weight(
                shape=(1, self._out_dims),
                initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                trainable=False,
                name="sn_u",
                dtype=self.w.dtype,
            )

    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            self.normalize_weights()

        output = self.layer(inputs, training=training)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    def l2_normalize(self, x, axis):
        y = x / (tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keepdims=True)) + self._eps)
        return y
    
    def normalize_weights(self):
        """Generate spectral normalized weights.

        This method will update the value of `self.w` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """
        if self.conv:
            w = tf.reshape(self.w, [-1, self._num_out_channels])
        else:
            w = tf.reshape(self.w, [-1, self._out_dims])
            
        u = self.u

        with tf.name_scope("spectral_norm_constraint"):
            for _ in range(self.power_iterations):
                v = self.l2_normalize(tf.matmul(u, w, transpose_b=True), axis=-1)
                u = self.l2_normalize(tf.matmul(v, w), axis=-1)
            
            u = tf.stop_gradient(u)
            v = tf.stop_gradient(v)
            
            sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True) + self._eps
            sigma_clipped = tf.minimum(sigma, self.max_spectral_norm)
            
            self.u.assign(tf.cast(u, self.u.dtype))
            self.w.assign(
                tf.cast(tf.reshape(self.w * (sigma_clipped / sigma), self.w_shape), self.w.dtype)
            )

    def get_config(self):
        config = {
            "max_spectral_norm": self.max_spectral_norm,
            "conv": self.conv,
            "power_iterations": self.power_iterations,
            "eps": self._eps
        }
        base_config = super().get_config()
        
        return {**base_config, **config}