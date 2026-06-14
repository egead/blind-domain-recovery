import numpy as np
import tensorflow as tf

class DataGenerator2d:
    def __init__(
        self,
        batch_size,
        features,
        feature_type="mnist",
        latent_x_dims=7,
        latent_y_dims=7,
        noise_normalized_std=0.0,
        output_representation="natural",
        random_linear_tf_min_eval=0.75,
        random_linear_tf_max_eval=1.33,
        give_bittensor=False,
        flatten_output=True,
        eps=1e-6,
        **kwargs
    ):
        self.batch_size = batch_size
        self.features = features
        self.give_bittensor = give_bittensor
        self.feature_type = feature_type
        self.latent_x_dims = latent_x_dims
        self.latent_y_dims = latent_y_dims
        self.random_linear_tf_min_eval = random_linear_tf_min_eval
        self.random_linear_tf_max_eval = random_linear_tf_max_eval
        self.noise_normalized_std = noise_normalized_std
        self.output_representation = output_representation
        self.flatten_output = flatten_output
        self.eps = eps
        
        np.random.seed(0)
        
        if self.feature_type == "mnist":
            # Load MNIST
            (x_train_mnist, _), (x_test_mnist, _) = tf.keras.datasets.mnist.load_data()
            self._mnist_data = np.concatenate([x_train_mnist, x_test_mnist], axis=0)
            
        self.batch_counter = 0
    
    def reset_batch_counter(self):
        self.batch_counter = 0
        
    def sample_batch_of_data(self, return_hidden_signal=False):
        """
        Main method to sample a batch of data from the specified feature types. 
        """
        x = np.zeros(
            shape=[self.batch_size, self.latent_x_dims, self.latent_y_dims], 
            dtype=np.float32
        )
        
        for feature in self.features:
            if feature["type"] == "mnist":
                total_length = 0
                samples = []
                give_bittensor = feature.get("give_bittensor", False)
                
                while True:
                    x_sampled = self._lot_mnist_crops(
                        max_warping_amplitude = feature.get("max_warping_amplitude", 0.),
                        blackout_probability = feature.get("blackout_probability", 0.),
                        blackout_mask_seed = feature.get("blackout_mask_seed", 0),
                        shading_probability = feature.get("shading_probability", 0.),
                        shaded_pixel_relative_intensity = feature.get("shaded_pixel_relative_intensity", 1.),
                        high_pass_filter = feature.get("use_high_pass_filter", False),
                        high_pass_filter_sigma = feature.get("high_pass_filter_sigma", 10),
                        apply_padding = feature.get("apply_padding", True),
                        give_bittensor = give_bittensor
                    )
                    x_sampled = self._filter_blank_images(x_sampled)
                    samples.append(x_sampled)
                    total_length = total_length + len(x_sampled)
                    if total_length >= self.batch_size:
                        break
                    
                x = tf.concat(samples, axis=0)
                x = x[0: self.batch_size]
                x = tf.expand_dims(x, axis=-1)
                
                break       
            else:
                raise ValueError('Feature type is not defined.')
        
        x = self._add_noise(x)
        
        if self.give_bittensor:
            x_transformed = self._convert_grayscale_to_bittensor(x)
        else:
            x_transformed = x
            
        if self.output_representation == "natural":
            x_transformed = x_transformed
        elif self.output_representation == "permuted":
            x_transformed = self._apply_permutation_map(x_transformed)
        elif self.output_representation == "dst":
            x_transformed = self._apply_dst_transform(x_transformed)
        elif self.output_representation == "linear":
            x_transformed = self._apply_random_linear_transform(x_transformed)      
        else:
            raise ValueError('Unsupported representation type.')
                
        if self.flatten_output:
            x_transformed = tf.reshape(x_transformed, 
            shape=[self.batch_size, -1])
                
        self.batch_counter += 1
        
        if return_hidden_signal:
            return x_transformed, x
        else:
            return x_transformed
    
    def _apply_high_pass_filter(self, x, sigma=10.0):
        """
        Removes DC component via Gaussian blur subtraction (Unsharp Masking).
        Preserves translation symmetry.
        """
        x = tf.cast(x, tf.float32)
        
        if len(x.shape) == 3:
            x_input = tf.expand_dims(x, axis=-1)
        else:
            x_input = x

        channels = tf.shape(x_input)[-1]
        
        kernel_size = int(4 * sigma) | 1 
        kernel = self._create_gaussian_kernel(sigma, kernel_size)
        
        kernel = tf.tile(kernel, [1, 1, channels, 1])
        
        blurred = tf.nn.depthwise_conv2d(
            x_input, kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        
        high_pass = x_input - blurred
        
        if len(x.shape) == 3:
            high_pass = tf.squeeze(high_pass, axis=-1)
        
        high_pass = tf.cast(high_pass + 0.5, tf.uint8)
            
        return high_pass
    
    def _create_gaussian_kernel(self, sigma, kernel_size):
        """Creates a 2D Gaussian kernel."""
        x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
        y = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
        x_grid, y_grid = tf.meshgrid(x, y)
        
        kernel = tf.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        kernel = kernel / tf.reduce_sum(kernel)
        return kernel[:, :, tf.newaxis, tf.newaxis] 
    
    def _lot_mnist_crops(self, 
                         apply_padding=True, 
                         blackout_probability=0.0, 
                         blackout_mask_seed=0, 
                         shading_probability=0.0,
                         shading_mask_seed=0,
                         shaded_pixel_relative_intensity=1.0,
                         high_pass_filter=False,
                         high_pass_filter_sigma=10,
                         **kwargs):
        images = self._sample_batch(self._mnist_data)
        
        if high_pass_filter:
            images = self._apply_high_pass_filter(images, high_pass_filter_sigma)
        
        cropped_images = self._randomly_crop_images(images=images, 
                                                    H_crop_size=self.latent_x_dims, 
                                                    W_crop_size=self.latent_y_dims, 
                                                    apply_padding=apply_padding)
        
        cropped_images = tf.reshape(
            cropped_images,
            [self.batch_size, self.latent_x_dims, self.latent_y_dims]
        )
        
        if blackout_probability > self.eps:
            cropped_images = self._apply_pixel_blackout(images=cropped_images, 
                                                        blackout_probability=blackout_probability, 
                                                        blackout_seed=blackout_mask_seed)
        
        if shading_probability > self.eps:
            cropped_images = self._apply_pixel_shading(images=cropped_images, 
                                                       shading_probability=shading_probability, 
                                                       shaded_pixel_relative_intensity=shaded_pixel_relative_intensity,
                                                       shading_seed=shading_mask_seed)
                    
        return cropped_images
    
    def _randomly_crop_images(self, images, H_crop_size, W_crop_size, apply_padding=False, only_center_crop=True):
        shape = tf.shape(images)
        batch_size = shape[0]
        H = shape[1]
        W = shape[2]
        
        if apply_padding:
            H = H + 2 * H_crop_size
            W = W + 2 * W_crop_size
            
            images = tf.image.pad_to_bounding_box(
                images,
                offset_height = H_crop_size,
                offset_width = W_crop_size,
                target_height = H,
                target_width = W
            )
            
        height_ratio = tf.cast(H_crop_size, tf.float32) / tf.cast(H, tf.float32)
        width_ratio  = tf.cast(W_crop_size, tf.float32) / tf.cast(W, tf.float32)
        
        max_y_start = 1.0 - height_ratio
        max_x_start = 1.0 - width_ratio

        if only_center_crop:
            y1 = tf.fill([batch_size], max_y_start / 2.0)
            x1 = tf.fill([batch_size], max_x_start / 2.0)
        else:
            y1 = tf.random.uniform([batch_size], minval=0.0, maxval=max_y_start)
            x1 = tf.random.uniform([batch_size], minval=0.0, maxval=max_x_start)

        y2 = y1 + height_ratio
        x2 = x1 + width_ratio
        
        boxes = tf.stack([y1, x1, y2, x2], axis=1)
        
        box_indices = tf.range(batch_size)
        crop_size   = [H_crop_size, W_crop_size]
        
        images_cropped = tf.image.crop_and_resize(
            images,
            boxes,
            box_indices,
            crop_size
        )
        
        return images_cropped
    
    def _apply_pixel_blackout(self, images, blackout_probability, blackout_seed):
        H = tf.shape(images)[1]
        W = tf.shape(images)[2]
        
        blackout = tf.random.uniform(shape=[H, W], 
                                     seed=blackout_seed) < blackout_probability
        mask = tf.where(blackout, 0., 1)
        images = images * mask[None, :, :]
        
        return images
    
    def _apply_pixel_shading(self, images, shading_probability, shaded_pixel_relative_intensity, shading_seed):
        H = tf.shape(images)[1]
        W = tf.shape(images)[2]
        
        blackout = tf.random.uniform(shape=[H, W], 
                                     seed=shading_seed) < shading_probability
        
        mask = tf.where(blackout, 0., 1)
        images = images * (mask[None, :, :] + shaded_pixel_relative_intensity * (1. - mask[None, :, :]))
        
        return images
    
    def _sample_batch(self, data):
        random_indices = tf.random.uniform(
            shape=[self.batch_size],
            minval=0,
            maxval=len(data),
            dtype=tf.int32,
            seed=0
        )
        
        samples = tf.gather(data, random_indices)
        samples = tf.expand_dims(samples, axis=-1)

        return samples

    def _filter_blank_images(self, x_images):
        image_maximums = tf.reduce_max(x_images, axis=(1, 2))
        mask = image_maximums > self.eps
        x_images = tf.boolean_mask(x_images, mask)
    
        return x_images
    
    def _add_noise(self, x_batch):
        x_batch_std = tf.math.reduce_std(x_batch, keepdims=True)
        noise = (
            self.noise_normalized_std
            * x_batch_std
            * tf.random.normal(mean=0.0, stddev=1., shape=tf.shape(x_batch), dtype=x_batch.dtype)
        )
        return x_batch + noise

    def _apply_permutation_map(self, x):
        num_c = tf.shape(x)[-1]
        x = tf.reshape(x, shape=[self.batch_size, self.latent_x_dims * self.latent_y_dims * num_c])
        x = np.matmul(x, self.permutation_matrix)
        x = tf.reshape(x, shape=[self.batch_size, self.latent_x_dims, self.latent_y_dims, num_c])
        return x

    def _convert_grayscale_to_bittensor(self, img):
        """
        img: uint8 numpy array or tensor of shape (..., H, W) 
            (Can handle Batched (B, H, W) or Single (H, W))
        
        returns: Float32 tensor of shape (..., H, W, 8)
        
        Output bits are ordered MSB -> LSB.
        """
        img = tf.cast(img, tf.uint8)
        img_expanded = tf.expand_dims(img, axis=-1) 

        shifts = tf.cast(tf.range(7, -1, -1, dtype=tf.int32), tf.uint8)
        masks = tf.bitwise.left_shift(tf.cast(1, tf.uint8), shifts)

        masked = tf.bitwise.bitwise_and(img_expanded, masks)
        bit_tensor = tf.where(masked > 0, 1.0, 0.0)

        return bit_tensor

    def _apply_dst_transform(self, x):
        x = tf.einsum("bijc, il, jk->blkc", 
                      x, 
                      self.dst_matrix_x, 
                      self.dst_matrix_y)
        return x
    
    def _apply_random_linear_transform(self, x):
        xs = x.shape
        x_flat = np.reshape(x, newshape=[xs[0], -1])
        x_flat_modified = x_flat @ self.random_linear_matrix
        x_modified = np.reshape(x_flat_modified, newshape=[xs[0], xs[1], xs[2], xs[3]])
        return x_modified
    
    # ----------------------------------------------------------------
    # Analysis & Metrics
    # ----------------------------------------------------------------    
    @property
    def random_linear_matrix(self):
        np.random.seed(0)
        d = self.latent_x_dims * self.latent_y_dims

        L = np.random.normal(size=(d, d))
        U, S, Vh = np.linalg.svd(L)

        a = (self.random_linear_tf_max_eval - self.random_linear_tf_min_eval) / (np.max(S) - np.min(S))
        b = self.random_linear_tf_min_eval - a * np.min(S)
        
        S = a * S + b
        L = U @ np.diag(S) @ Vh
        
        return L
    
    @property
    def permutation_matrix(self, seed=0):
        np.random.seed(seed)
        if self.give_bittensor:
            out_size = 8 * self.latent_x_dims * self.latent_y_dims
        else:
            out_size = self.latent_x_dims * self.latent_y_dims
            
        perm = np.random.permutation(np.arange(out_size))
        v = np.concatenate(
            [
                np.ones([1, 1], dtype=np.float32),
                np.zeros([1, out_size - 1], dtype=np.float32),
            ],
            axis=1,
        )

        v_translated = []
        for p in perm:
            v_translated.append(np.roll(v, axis=1, shift=p))

        perm = np.concatenate(v_translated, axis=0)
        return perm
    
    @property
    def dst_matrix_x(self):
        ts_float = np.array(self.latent_x_dims, dtype=np.float32)
        n = np.arange(1, self.latent_x_dims + 1, dtype=np.float32)[:, None]
        k = np.arange(1, self.latent_x_dims + 1, dtype=np.float32)[None, :]
        dst = np.sin(n * k * np.pi / (ts_float + 1.0))
        return dst / np.sqrt((self.latent_x_dims + 1) / 2)

    @property
    def dst_matrix_y(self):
        ts_float = np.array(self.latent_y_dims, dtype=np.float32)
        n = np.arange(1, self.latent_y_dims + 1, dtype=np.float32)[:, None]
        k = np.arange(1, self.latent_y_dims + 1, dtype=np.float32)[None, :]
        dst = np.sin(n * k * np.pi / (ts_float + 1.0))
        return dst / np.sqrt((self.latent_y_dims + 1) / 2)