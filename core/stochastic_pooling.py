import tensorflow as tf


class StochasticPooling2(tf.keras.layers.Layer):
    """
    Stochastic-Pooling (Zeiler & Fergus, 2013).
    """

    def __init__(self,
                 strides: int = 2,
                 padding: str = "VALID",
                 epsilon: float = 1e-8,
                 **kwargs):
        super().__init__(**kwargs)
        self.pool_size = 2            # sabit!
        self.strides = int(strides)
        self.padding = padding.upper()
        self.epsilon = float(epsilon)

    def _extract_pairs(self, x):
        """[B, L, C] → [B, L_out, C, 2] pencerelerini çeker."""
        # [B, L, C]  ->  [B, L, 1, C]  (fake width)
        x = tf.expand_dims(x, 2)

        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.pool_size, 1, 1],          # 2‑li pencereler
            strides=[1, self.strides, 1, 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )
        B  = tf.shape(patches)[0]
        Lo = tf.shape(patches)[1]
        C  = x.shape[-1]                # statik int (kanal sayısı)
        return tf.reshape(patches, [B, Lo, C, 2])      # son eksen = 2’lik pencere

    # ――― Ana ――― #
    def call(self, inputs, training=None):
        pairs = self._extract_pairs(inputs)             # [B, Lo, C, 2]

        pos = tf.nn.relu(pairs) + self.epsilon
        s   = tf.reduce_sum(pos, axis=-1, keepdims=True)

        p0  = pos[..., 0] / s[..., 0]                   # a₀ / (a₀ + a₁)

        if training:
            # Bernoulli(p0) → 0 (ilk) / 1 (ikinci) indeksi
            rnd  = tf.random.uniform(tf.shape(p0), dtype=p0.dtype)
            idx  = tf.cast(rnd > p0, tf.int32)          # rnd>p0 ⇒ 1 seçilir
            # tf.gather:  batch_dims=3 → ilk 3 eksen (B,Lo,C) toplu ele al
            out  = tf.gather(pairs, idx, axis=-1, batch_dims=3)    # [B, Lo, C]
        else:
            # Beklenen değer (det.)
            p1  = 1.0 - p0
            out = p0 * pairs[..., 0] + p1 * pairs[..., 1]

        return out

class StochasticPooling2x2(tf.keras.layers.Layer):
    """
    2D Stochastic Pooling (Zeiler & Fergus, 2013) - pencere=2x2 sabit.
    channels_last format: [B, H, W, C]
    """

    def __init__(self,
                 strides=(2, 2),
                 padding="VALID",
                 epsilon=1e-8,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(strides, int):
            strides = (strides, strides)
        self.pool_size = (2, 2)     # sabit
        self.strides = tuple(int(s) for s in strides)
        self.padding = padding.upper()
        self.epsilon = float(epsilon)

    def _extract_quads(self, x):
        xs = tf.shape(x)
        B, H, W, C = xs[0], xs[1], xs[2], xs[3]
        
        # Make H and W even.
        H = 2 * (H // 2)
        W = 2 * (W // 2)
        x = tf.slice(x, begin=[0, 0, 0, 0], size=[B, H, W, C])
        
        # B, H, W, C -> B, C, H, W
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        
        # B, C, H, W -> B, C, H, W//2, 2
        x = tf.reshape(x, shape=[B, C, H, W//2, 2])
        
        # B, C, H, W//2, 2 -> B, C, W//2, H, 2
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4])
        
        # B, C, H, W//2, 2 -> B, C, W//2, H//2, 4
        x = tf.reshape(x, shape=[B, C, W//2, H//2, 4])
        
        # B, C, W//2, H//2, 4 -> B, H // 2, W // 2, C, 4
        x = tf.transpose(x, perm=[0, 3, 2, 1, 4])
        
        return x
        
    def call(self, inputs, training=None):
        quads = self._extract_quads(inputs)             # [B, H_out, W_out, C, 4]

        pos = tf.nn.relu(quads) + self.epsilon
        sums = tf.reduce_sum(pos, axis=-1, keepdims=True)   # [B, H_out, W_out, C, 1]
        probs = pos / sums   

        if training:
            u = tf.random.uniform(tf.shape(probs)[:-1], dtype=probs.dtype)  # [B,H_out,W_out,C]
            cdf = tf.cumsum(probs, axis=-1)                                 # [B,H_out,W_out,C,4]
            
            greater = cdf > tf.expand_dims(u, axis=-1)                      # bool
            idx = tf.argmax(tf.cast(greater, tf.int32), axis=-1)            # [B,H_out,W_out,C]
             
            out = tf.gather(quads, idx, axis=-1, batch_dims=4)              # [B,H_out,W_out,C]
        else:
            out = tf.reduce_sum(probs * quads, axis=-1)                     # [B,H_out,W_out,C]

        return out

class StochasticPooling2x2x2(tf.keras.layers.Layer):
    def __init__(self,
                 padding="VALID",
                 epsilon=1e-8,
                 **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def _extract_octos(self, x):
        xs = tf.shape(x)
        B, H, W, Z, C = xs[0], xs[1], xs[2], xs[3], xs[4]
        
        # Make H, W, Z even.
        H = 2 * (H // 2)
        W = 2 * (W // 2)
        Z = 2 * (Z // 2)
        x = tf.slice(x, begin=[0, 0, 0, 0, 0], size=[B, H, W, Z, C])
        
        x = tf.transpose(x, [0, 4, 1, 2, 3])                  # B,C,H,W,Z
        x = tf.reshape(x, [B, C, H//2, 2, W//2, 2, Z//2, 2])         # split dims
        x = tf.transpose(x, [0, 2, 4, 6, 1, 3, 5, 7])               # B,H//2,W//2,Z//2,C,2,2,2
        x = tf.reshape(x, [B, H//2, W//2, Z//2, C, 8])            # final octos
                
        return x
        
    def call(self, inputs, training=None):
        octos = self._extract_octos(inputs)             # [B, H_out, W_out, Z_out, C, 8]

        pos = tf.nn.relu(octos) + self.epsilon
        sums = tf.reduce_sum(pos, axis=-1, keepdims=True)   # [B, H_out, W_out, Z_out, C, 1]
        probs = pos / sums

        if training:
            u = tf.random.uniform(tf.shape(probs)[:-1], dtype=probs.dtype)  # [B,H_out,W_out,Z_out, C]
            cdf = tf.cumsum(probs, axis=-1)                                 # [B,H_out,W_out,Z_out, C,8]

            greater = cdf > tf.expand_dims(u, axis=-1)                      # bool
            idx = tf.argmax(tf.cast(greater, tf.int32), axis=-1)            # [B,H_out,W_out,Z_out,C]
             
            out = tf.gather(octos, idx, axis=-1, batch_dims=5)              # [B,H_out,W_out,Z_out,C]
        else:
            out = tf.reduce_sum(probs * octos, axis=-1)                     # [B,H_out,W_out,Z_out, C]

        return out