import tensorflow as tf
import numpy as np

class OrthogonalConstraint(tf.keras.constraints.Constraint):
    """Symmetric orthogonalization for W."""
    def __call__(self, w):
        wt = tf.transpose(w) 
        wtw = tf.matmul(wt, w) 
        e, v = tf.linalg.eigh(wtw)
        e_inv_sqrt = tf.linalg.diag(tf.math.rsqrt(e + 1e-8))
        wtw_inv_sqrt = tf.matmul(v, tf.matmul(e_inv_sqrt, tf.transpose(v)))
        return tf.matmul(w, wtw_inv_sqrt)

class TICAModel1D(tf.keras.Model):
    def __init__(self, 
                 input_dim, 
                 n_components=63,
                 pool_size=3, 
                 det_penalty=1e-1,
                 epsilon=1e-6, 
                 use_orthogonal=True,
                 is_circulant=True, # Recommended True for cyclic translation datasets
                 **kwargs):
        super(TICAModel1D, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.use_orthogonal = use_orthogonal
        self.n_components = n_components
        self.det_penalty = det_penalty
        self.output_dim = self.n_components
        
        constraint = OrthogonalConstraint() if self.use_orthogonal else None
        
        self.W = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer='orthogonal',
            trainable=True,
            constraint=constraint,
            name='tica_weights'
        )
        
        # Build the 1D neighborhood pooling matrix
        v_matrix = np.zeros((self.output_dim, self.output_dim), dtype=np.float32)
        radius = pool_size // 2 

        for i in range(self.n_components):
            for d in range(-radius, radius + 1):
                neighbor_idx = i + d
                
                if is_circulant:
                    # Wrap around (periodic boundary conditions)
                    neighbor_idx = neighbor_idx % self.n_components
                    v_matrix[i, neighbor_idx] = 1.0
                else:
                    # Hard boundaries (no wrap-around)
                    if 0 <= neighbor_idx < self.n_components:
                        v_matrix[i, neighbor_idx] = 1.0

        self.V = tf.constant(v_matrix)

    def call(self, x):
        if self.use_orthogonal:
            w_current = self.W
        else:
            w_current = tf.math.l2_normalize(self.W, axis=0)
            
        # 1. Compute the flat output (shape: batch_size, output_dim)
        y = tf.matmul(x, w_current)
        
        # 2. Compute Loss (Requires flat y)
        y_squared = tf.square(y)
        pooled_energies = tf.matmul(y_squared, self.V, transpose_b=True)
        
        topographic_loss = tf.reduce_mean(tf.sqrt(pooled_energies + self.epsilon))
        
        if self.use_orthogonal:
            total_loss = topographic_loss
        else:
            wtw = tf.matmul(w_current, w_current, transpose_a=True)
            
            # Use output_dim to match the actual number of learned components
            eye = tf.eye(self.output_dim) 
            wtw_stable = wtw + (eye * 1e-6)
            
            _, log_det = tf.linalg.slogdet(wtw_stable)
            
            det_penalty = -0.5 * log_det 
            
            total_loss = topographic_loss + (self.det_penalty * tf.reduce_mean(det_penalty))

        self.add_loss(total_loss)
        
        # 3. Reshape to 1D Signal Format
        # -1 tells TensorFlow to infer the batch size automatically
        y_signal = tf.reshape(y, (-1, self.n_components))
        
        return y_signal