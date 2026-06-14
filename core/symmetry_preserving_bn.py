import tensorflow as tf

class SymmetryPreservingBatchNorm(tf.keras.layers.Layer):
    def __init__(self, momentum=0.9, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        # We use standard BatchNormalization, but we will force it to see 1 channel
        self.bn = tf.keras.layers.BatchNormalization(
            axis=-1, # Normalize along the last axis (the dummy channel 1)
            momentum=momentum,
            epsilon=epsilon,
            center=False,
            scale=False
        )

    def call(self, x, training=False):
        # x shape: [Batch, Features] (e.g., [32, 784])
        
        # 1. Add a dummy channel dimension
        # Shape becomes [Batch, Features, 1]
        # Keras BN treats 'Features' as the "Time/Space" dimension and '1' as the Channel.
        # It aggregates stats over (Batch + Features), producing 1 scalar mean/var.
        x_expanded = tf.expand_dims(x, axis=-1)
        
        # 2. Apply BN
        out = self.bn(x_expanded, training=training)
        
        # 3. Remove the dummy dimension to return to [Batch, Features]
        return tf.squeeze(out, axis=-1)