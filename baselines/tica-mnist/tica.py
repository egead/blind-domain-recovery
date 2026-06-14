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

class TICAModel(tf.keras.Model):
    def __init__(self, 
                 input_dim, 
                 grid_x_size=15,
                 grid_y_size=15,
                 pool_size=3, 
                 det_penalty=1e-1,
                 epsilon=1e-6, 
                 use_orthogonal=True, 
                 **kwargs):
        super(TICAModel, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.use_orthogonal = use_orthogonal
        self.grid_x_size = grid_x_size
        self.grid_y_size = grid_y_size
        self.det_penalty = det_penalty
        self.output_dim = self.grid_x_size * self.grid_y_size
        
        constraint = OrthogonalConstraint() if self.use_orthogonal else None
        
        self.W = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer='orthogonal',
            trainable=True,
            constraint=constraint,
            name='tica_weights'
        )
        
        # Build the 2D neighborhood pooling matrix
        v_matrix = np.zeros((self.output_dim, self.output_dim), dtype=np.float32)
        radius = pool_size // 2 

        for row in range(self.grid_x_size):
            for col in range(self.grid_y_size):
                # Calculate the flat index for the central unit using row-major order
                input_linear_index = row * self.grid_y_size + col
                
                # Iterate over the 2D neighborhood
                for d_row in range(-radius, radius + 1):
                    for d_col in range(-radius, radius + 1):
                        n_row = row + d_row
                        n_col = col + d_col
                        
                        # Hard boundaries (no wrap-around)
                        if 0 <= n_row < self.grid_x_size and 0 <= n_col < self.grid_y_size:
                            output_linear_index = n_row * self.grid_y_size + n_col
                            v_matrix[input_linear_index, output_linear_index] = 1.0

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
        
        # 3. Reshape to 2D Image Format
        # -1 tells TensorFlow to infer the batch size automatically
        y_image = tf.reshape(y, (-1, self.grid_x_size, self.grid_y_size))
        
        return y_image