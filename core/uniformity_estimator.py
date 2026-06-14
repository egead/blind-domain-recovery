import tensorflow as tf
from keras.layers import (Convolution1D, 
                          Convolution2D, 
                          Convolution3D, 
                          MaxPool1D, 
                          MaxPool2D,
                          MaxPool3D,
                          GlobalAveragePooling1D,
                          GlobalAveragePooling2D,
                          GlobalAveragePooling3D)
from core.spectral_norm_constraint import SpectralNormConstraint
from core.stochastic_pooling import StochasticPooling2, StochasticPooling2x2, StochasticPooling2x2x2
from core.regularizations import convert_to_regularization_format
from copy import deepcopy

@tf.keras.utils.register_keras_serializable()
class UniformityEstimator(tf.keras.models.Model):
    def __init__(self, 
                 layer_properties: list = None,
                 name: str = "uniformity_estimator",
                 group_dims: int = 1,
                 embedding_dim: int = 4,
                 use_input_embedding: bool = True,
                 use_stochastic_pooling: bool = True,
                 max_spectral_norm: float = 10.0,
                 nn_output_clamp: float = 25.0,
                 embedding_clipvalue: float = 5.0,
                 use_final_gap_layer: bool = True,
                 cyclic_boundary: bool = False, 
                 eps: float = 1e-7):
        super().__init__(name=name)          
        self._layer_properties = layer_properties
        self._group_dims = group_dims
        self._embedding_dim = embedding_dim
        self._use_final_gap_layer = use_final_gap_layer
        self._use_input_embedding = use_input_embedding
        self._embedding_clipvalue = embedding_clipvalue
        self._use_stochastic_pooling = use_stochastic_pooling
        self._nn_output_clamp = nn_output_clamp
        self._cyclic_boundary = cyclic_boundary
        self._max_spectral_norm = max_spectral_norm
        self._eps = eps

    def get_config(self):
        config = super(UniformityEstimator, self).get_config()
        config.update({
            "layer_properties": self._layer_properties,
            "group_dims": self._group_dims,
            "embedding_dim": self._embedding_dim,
            "use_input_embedding": self._use_input_embedding,
            "use_stochastic_pooling": self._use_stochastic_pooling,
            "max_spectral_norm": self._max_spectral_norm,
            "nn_output_clamp": self._nn_output_clamp,
            "embedding_clipvalue": self._embedding_clipvalue,
            "use_final_gap_layer": self._use_final_gap_layer,
            "cyclic_boundary": self._cyclic_boundary,
            "eps": self._eps,
        })
        return config
            
    def build(self, input_shape=None):
        self._input_shape = input_shape
        self._create_networks()
        
    def call(self, inputs, training=False):
        x = inputs
        x = self._precondition(x)
        
        js_distance_squared = []
        regularization = []
        
        for dim_idx in range(self._group_dims):
            px, qx = self._split_along_translation(x, dim_idx, 1)
            
            js_div, reg = self._fit_js_divergence(px,
                                                  qx,  
                                                  dim_idx=dim_idx, 
                                                  training=training)
            
            js_per_dim = self._compute_js_div_per_dim(js_div, dim_idx)
                
            Ts = self._compute_sampling_interval(dim_idx)
            js_distance_squared.append(js_per_dim / (tf.square(Ts)))
            regularization.append(tf.reduce_sum(reg))
                    
        js_distance_squared = tf.stack(js_distance_squared, axis=0)
        regularization = tf.stack(regularization, axis=0)
            
        uniformity = tf.reduce_mean(js_distance_squared)
        total_regularization = tf.reduce_sum(regularization)
        
        # Add loss term.
        if training:
            self._add_regularization_terms(total_regularization)
        
        return uniformity
    
    def _create_networks(self):
        self._networks = []
        
        if self._use_input_embedding:
            self._embeddings = []
        
        for dim_idx in range(self._group_dims):
            for direction_idx in range(self._num_directions):
                if self._use_input_embedding:
                    embedding_name = self._get_embedding_name(dim_idx, direction_idx)
                    embedding = self._create_input_embedding(dim_idx, 
                                                             name=embedding_name)
                    self._embeddings.append(embedding)
                
                network_name = self._get_network_name(dim_idx, direction_idx)
                network = self._create_cnn_network(self._layer_properties, 
                                                   name=network_name)
                
                self._networks.append(network)

    def _create_cnn_network(self, layer_properties, name):
        layers = []
        
        layer_max_spectral_norm = self._max_spectral_norm / len(layer_properties)
        
        if self._group_dims == 1:
            if self._use_stochastic_pooling:
                pooling_ctor = StochasticPooling2
            else:
                pooling_ctor = MaxPool1D
                
            gap_ctor = GlobalAveragePooling1D
            convolution_ctor = Convolution1D
        elif self._group_dims == 2:
            if self._use_stochastic_pooling:
                pooling_ctor = StochasticPooling2x2
            else:
                pooling_ctor = MaxPool2D
                
            convolution_ctor = Convolution2D
            gap_ctor = GlobalAveragePooling2D
        elif self._group_dims == 3:
            if self._use_stochastic_pooling:
                pooling_ctor = StochasticPooling2x2x2
            else:
                pooling_ctor = MaxPool3D
                
            convolution_ctor = Convolution3D
            gap_ctor = GlobalAveragePooling3D
        else:
            raise ValueError("Group dimensionalities higher than 3D is not implemented.")
            
        for layer_idx, layer_property in enumerate(layer_properties):
            layer_property = deepcopy(layer_property)
            if "pooling" in layer_property.keys():
                pooling = layer_property["pooling"]
                layer_property.pop("pooling")
            else:
                pooling = False
                
            if layer_idx == (len(layer_properties) - 1):
                layer_property["activation"] = layer_property.get("activation", "linear")  
            else:
                layer_property["activation"] = layer_property.get("activation", "elu")
            
            layer = SpectralNormConstraint(convolution_ctor(padding="same", 
                                                            **layer_property), 
                                           conv=True,
                                           max_spectral_norm=layer_max_spectral_norm,
                                           name=f"{name}_sconstraint_{layer_idx}")
            layers.append(layer)
            
            if pooling:
                layers.append(pooling_ctor())
        
        if self._use_final_gap_layer:
            layers.append(gap_ctor())
                
        return layers
                            
    def _create_input_embedding(self, dim_idx, name):
        sample_shape = tf.convert_to_tensor(list(self._input_shape[1:]))
        dims = tf.range(0, len(sample_shape))
        
        if self._cyclic_boundary:
            embedding_shape = sample_shape
        else:
            embedding_shape = tf.where(dims == dim_idx, 
                                       sample_shape - 1, 
                                       sample_shape)
            
        embedding_shape = tf.concat([embedding_shape[:-1], 
                                     tf.convert_to_tensor([self._embedding_dim], dtype=tf.int32)], 
                                    axis=0)
        
        init_val = tf.zeros(shape=embedding_shape, dtype=tf.float32)
        embedding = tf.Variable(initial_value=init_val, trainable=True, name=name)

        return embedding
    
    def _add_regularization_terms(self, total_regularization):
        self.add_loss(convert_to_regularization_format(
            "uniformity_estimator_regularization", 
            total_regularization
        ))
    
    def _precondition(self, x):
        mu = tf.math.reduce_mean(x, keepdims=True)
        std = tf.sqrt(tf.reduce_mean(tf.math.reduce_variance(x, axis=0, keepdims=True)))
        x = (x - mu) / (std + self._eps)
        
        return x
    
    def _fit_js_divergence(self, px, qx, dim_idx, training=False):        
        N = tf.shape(px)[0]
        mx = tf.concat([px[:N//2], qx[N//2:]], axis=0)
        
        kl_pm, reg_pm = self._fit_kl_divergence(px, 
                                                mx,
                                                dim_idx=dim_idx, 
                                                direction_idx=0, 
                                                training=training)
        
        kl_qm, reg_qm = self._fit_kl_divergence(qx, 
                                                mx, 
                                                dim_idx=dim_idx,
                                                direction_idx=1, 
                                                training=training)
        
        js_div = 0.5 * (kl_pm + kl_qm)
        regularization = 0.5 * (reg_pm + reg_qm)
        
        return js_div, regularization
        
    def _fit_kl_divergence(self, px, qx, dim_idx, direction_idx, training=False):
        py = self._fit_nn(px, dim_idx, direction_idx, training=training)
        qy = self._fit_nn(qx, dim_idx, direction_idx, training=training)
        
        kl1 = self._compute_kl_divergence(py, qy)
        kl2 = self._compute_kl_divergence(qy, py)
        
        kl_div = tf.nn.relu(tf.maximum(kl1, kl2))
        regularization = -(tf.maximum(kl1, kl2))
        
        return kl_div, regularization
    
    def _fit_nn(self, x, dim_idx, direction_idx, training=False):
        network = self._get_network(dim_idx, direction_idx)
        
        if self._use_input_embedding:      
            embedding = self._get_embedding(dim_idx, direction_idx)
            n_repeats = self._embedding_dim // tf.shape(x)[-1]
            embedding = self._embedding_clipvalue * tf.nn.tanh(embedding / self._embedding_clipvalue)
            y = tf.repeat(x, axis=-1, repeats=n_repeats) + embedding[None, ...]
        else:
            y = x
            
        for layer in network:
            y = layer(y, training=training)
        
        ys = tf.shape(y)
        y = tf.reshape(y, shape=[ys[0], -1])
        
        y = self._nn_output_clamp * tf.nn.tanh(y / self._nn_output_clamp)
        return y
        
    def _split_along_translation(self, x, dim_idx, translation):
        if self._cyclic_boundary:
            px = x
            qx = tf.roll(x, shift=-translation, axis=(dim_idx+1))
        else:
            px = self._crop(x, dim_idx=dim_idx, size=translation, crop_end=True)
            qx = self._crop(x, dim_idx=dim_idx, size=translation, crop_end=False)
        
        return px, qx
    
    def _crop(self, x, dim_idx, size, crop_end=True):
        xshape = tf.shape(x)
        xrank = tf.rank(x)
        
        axes = tf.range(xrank)
        
        begins = tf.where((axes == (dim_idx + 1)) & (not crop_end), size, 0)
        sizes = tf.where((axes == (dim_idx + 1)), xshape-size, xshape)
        
        x = tf.slice(x, begin=begins, size=sizes)
        return x
    
    def _compute_kl_divergence(self, py, qy):        
        mu_y = tf.reduce_mean(py, axis=0, keepdims=True)
        diff = qy - mu_y
        
        kl_div = -(tf.math.log(tf.reduce_mean(tf.exp(diff), axis=0) + self._eps))
        
        return kl_div
    
    def _get_embedding(self, dim_idx, direction_idx):
        idx = (dim_idx * self._num_directions + 
               direction_idx)
        
        embedding = self._embeddings[idx]
        return embedding
    
    def _get_network(self, dim_idx, direction_idx):
        idx = (dim_idx * self._num_directions + 
               direction_idx)
        
        network = self._networks[idx]
        return network
    
    def _compute_sampling_interval(self, dim_idx):
        dilation_f = tf.cast(1.0, tf.float32)
        receptive_field_f = tf.cast(1.0, tf.float32)
        n_f = tf.cast(self._input_shape[dim_idx+1], tf.float32)
        Ts = receptive_field_f / (n_f / dilation_f)
        return Ts
    
    def _compute_js_div_per_dim(self, js_div, dim_idx):
        dims_f = tf.cast(self._input_shape[1:-1], tf.float32)
        
        n_axes_dims_f = dims_f[dim_idx]
        n_other_dims_f = tf.reduce_prod(dims_f) / n_axes_dims_f
        
        if self._cyclic_boundary:
            d_f = n_other_dims_f * n_axes_dims_f
        else:
            translation_f = tf.cast(1.0, tf.float32)
            d_f = n_other_dims_f * (n_axes_dims_f - translation_f)

        return (js_div / d_f)

    def _get_embedding_name(self, dim_idx, direction_idx):
        return f"embedding_dim{dim_idx}_dir{direction_idx}"
    
    def _get_network_name(self, dim_idx, direction_idx):
        return f"cnn_network_dim{dim_idx}_dir{direction_idx}"
    
    @staticmethod
    def _to_tensor(x, dtype=tf.int32):
        return tf.convert_to_tensor(x, dtype=dtype)
    
    @property
    def _num_directions(self):
        return 2