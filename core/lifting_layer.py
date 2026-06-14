import math
import tensorflow as tf
from tensorflow.linalg import eigh


@tf.keras.utils.register_keras_serializable()
class LiftingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        axes_widths,
        num_resolution_filters=1,
        resolution_filter_sigma_decay_tc_in_epochs=10.0,
        resolution_filter_initial_sigma=0.1,
        steps_per_epoch=100,
        safe_eigh_shift=1e-5,
        use_spectral_lifting=True,
        use_zero_padded_filter=False,
        normalize_lifting_map=False,
        lifting_offset_limit=None,
        name="lifting_layer",
        eps=1e-7,
        *args,
        **kwargs
    ):
        self.axes_widths = axes_widths
        self.group_dims = len(axes_widths)
        self.num_resolution_filters = num_resolution_filters
        self.use_spectral_lifting = use_spectral_lifting
        self.use_zero_padded_filter = use_zero_padded_filter
        self.normalize_lifting_map = normalize_lifting_map
        self.resolution_filter_sigma_decay_tc_in_epochs = resolution_filter_sigma_decay_tc_in_epochs
        self.resolution_filter_initial_sigma = resolution_filter_initial_sigma
        self.steps_per_epoch = steps_per_epoch
        self.lifting_offset_limit = lifting_offset_limit
        self.safe_eigh_shift = safe_eigh_shift
        self.eps = eps
        super(LiftingLayer, self).__init__(name=name)

    def get_config(self):
        config = super(LiftingLayer, self).get_config()
        config.update({
            'axes_widths': self.axes_widths,
            'num_resolution_filters': self.num_resolution_filters,
            'safe_eigh_shift': self.safe_eigh_shift,
            'resolution_filter_sigma_decay_tc_in_epochs': self.resolution_filter_sigma_decay_tc_in_epochs,
            'resolution_filter_initial_sigma': self.resolution_filter_initial_sigma,
            'steps_per_epoch': self.steps_per_epoch,
            'use_spectral_lifting': self.use_spectral_lifting,
            'use_zero_padded_filter': self.use_zero_padded_filter,
            'lifting_offset_limit': self.lifting_offset_limit,
            'normalize_lifting_map': self.normalize_lifting_map,
            'eps': self.eps,
        })
        return config

    def build(self, input_shape=None):
        self._input_shape = input_shape
        self._create_step_counter()
        self._create_resolution_filters()
        self._create_eigenvector_parametrization()
        if self.lifting_offset_limit is not None:
            self._create_lifting_offsets()
        if self.group_dims > 1:
            self._create_eigenvalue_parametrizations()

    def call(self, x, training=False, analyze=False):
        if analyze:
            L = self.compute_lifting_map(training=False)
            x_flat = tf.reshape(x, [tf.shape(x)[0], -1])
            y = x_flat @ L
            y_shape = [-1] + [self.num_lifted_channels] + self.axes_widths
            y = tf.reshape(y, shape=y_shape)
            y = tf.einsum("bc...->b...c", y)
            return y, L

        if self.use_spectral_lifting:
            y = self.lift_spectral(x, training=training)
        else:
            y = self.lift(x, training=training)

        if training:
            self._increase_step_counter()

        return y

    def lift(self, x, training=False):
        x = tf.reshape(x, shape=[tf.shape(x)[0], -1])
        L = self.compute_lifting_map(training=training)
        y = x @ L
        y_shape = [-1] + [self.num_lifted_channels] + self.axes_widths
        y = tf.reshape(y, shape=y_shape)
        y = tf.einsum("bc...->b...c", y)
        return y

    def lift_spectral(self, x, training=False):
        if training:
            rf = self._sample_resolution_filters()
        else:
            rf = self._l2_normalize(self.resolution_filters, axis=-1)

        evecs, evals = self._decompose_generator_parametrization(self._eigenvector_parametrization)

        D = self.n_input_dims
        if self.use_zero_padded_filter:
            total_pad_size = D - tf.shape(rf)[1]
            if total_pad_size > 0:
                left_pad = total_pad_size // 2
                right_pad = total_pad_size - left_pad
                rf = tf.pad(rf, paddings=[[0, 0], [left_pad, right_pad]])

        rf_nw = self._map_to_generator_basis(rf, evecs)
        x_nw = self._map_to_generator_basis(x, evecs)

        rf_nw = tf.expand_dims(rf_nw, axis=0)
        x_nw = tf.expand_dims(x_nw, axis=1)
        f_nw = x_nw * tf.math.conj(rf_nw)

        I = self._compute_impedance_tensor(evecs, evals)

        # Flatten spatial dims for memory-efficient BatchMatMul
        I_flat = tf.reshape(I, [D, -1])
        y_flat = tf.math.real(tf.einsum("bcd, dm->bcm", f_nw, tf.math.conj(I_flat)))
        target_shape = tf.concat([tf.shape(y_flat)[:2], self.axes_widths], axis=0)
        y = tf.reshape(y_flat, target_shape)

        rank = tf.rank(y)
        perm = tf.concat([[0], tf.range(2, rank), [1]], axis=0)
        y = tf.transpose(y, perm=perm)

        return y

    def compute_lifting_map(self, training=False):
        D = self.n_input_dims

        if training:
            rf = self._sample_resolution_filters()
        else:
            rf = self._l2_normalize(self.resolution_filters, axis=-1)

        if self.use_zero_padded_filter:
            total_pad_size = D - tf.shape(rf)[1]
            if total_pad_size > 0:
                left_pad = total_pad_size // 2
                right_pad = total_pad_size - left_pad
                rf = tf.pad(rf, paddings=[[0, 0], [left_pad, right_pad]])

        evecs, evals = self._decompose_generator_parametrization(self._eigenvector_parametrization)

        rf_nw = self._map_to_generator_basis(rf, evecs)

        I = self._compute_impedance_tensor(evecs, evals)
        I_flat = tf.reshape(I, shape=[D, -1])

        Lw = tf.einsum("kl, ln->nkl", rf_nw, I_flat)
        L = self._retrieve_from_generator_basis(Lw, evecs)

        L = tf.transpose(L, perm=[2, 1, 0])
        L = tf.reshape(L, shape=[D, -1])

        if self.normalize_lifting_map:
            L = self._l2_normalize(L, axis=0)

        return L

    def _compute_impedance_tensor(self, evecs, evals):
        impedance_tensor = None

        for i in range(self.group_dims):
            if self.group_dims > 1:
                evals = self._compute_eigenvalues(self._eigenvalue_parametrizations[i], evecs)

            offset = 0.0
            if self.lifting_offset_limit is not None:
                offset = self.lifting_offset_limit * tf.nn.tanh(
                    self.lifting_offsets[i] / self.lifting_offset_limit
                )

            positions = tf.linspace(start=-0.5, stop=0.5, num=self.axes_widths[i])
            positions = positions + offset
            positions = positions * tf.cast(self.axes_widths[i] - 1, tf.float32)

            impedances = tf.exp(
                tf.einsum("l, n->ln", evals, self._complex(r=positions))
            )

            if impedance_tensor is None:
                impedance_tensor = impedances
            else:
                impedance_tensor = tf.einsum("l..., ln->l...n", impedance_tensor, impedances)

        return impedance_tensor

    def _decompose_generator_parametrization(self, generator_parametrization):
        s = self._skew_symmetrize(generator_parametrization)
        h = self._complex(i=s)
        eigvals, eigvecs = self._safe_eigh(h)
        eigvals = eigvals / self._complex(i=1.0)
        eigvals, eigvecs = self._sort_eigenvalues_and_eigenvectors(eigvals, eigvecs)
        return eigvecs, eigvals

    def _safe_eigh(self, m):
        n = tf.shape(m)[0]
        m_shifted = m + tf.eye(n, dtype=m.dtype) * self.safe_eigh_shift
        eigvals, eigvecs = eigh(m_shifted)
        return eigvals, eigvecs

    def _sort_eigenvalues_and_eigenvectors(self, eigenvalues, eigenvectors):
        phases = tf.math.imag(eigenvalues)
        sorter = tf.argsort(phases)
        sorted_eigenvalues = tf.gather(eigenvalues, sorter)
        sorted_eigenvectors = tf.gather(eigenvectors, sorter, axis=1)
        return sorted_eigenvalues, sorted_eigenvectors

    def _compute_eigenvalues(self, eigenvalue_param, eigenvectors):
        theta = tf.expand_dims(eigenvalue_param, axis=-1)
        C = tf.transpose(eigenvectors) @ eigenvectors
        C = tf.math.abs(C)
        idxs = tf.argmax(C, axis=-1)
        C = tf.one_hot(idxs, depth=tf.shape(C)[0], axis=-1)
        phases = theta - C @ theta
        return self._complex(i=phases[:, 0])

    def _sample_resolution_filters(self):
        sigma = self._compute_resolution_filters_sigma()
        noise = self._generate_resolution_filters_noise()
        filters = self.resolution_filters + sigma * noise
        filters = filters / tf.norm(filters, axis=1, keepdims=True)
        return filters

    def _generate_resolution_filters_noise(self):
        return tf.random.normal(shape=tf.shape(self.resolution_filters))

    def _compute_resolution_filters_sigma(self):
        t = tf.cast(self.step_counter, dtype=tf.float32) / self.steps_per_epoch
        sigma = self.resolution_filter_initial_sigma * tf.exp(
            -t / self.resolution_filter_sigma_decay_tc_in_epochs
        )
        return sigma

    def _increase_step_counter(self):
        incremented_value = self.step_counter + 1
        overflow_mask = tf.cast(incremented_value == 0, tf.uint32)
        next_value = (
            overflow_mask * self.step_counter + (1 - overflow_mask) * incremented_value
        )
        self.step_counter.assign(next_value)

    def _map_to_generator_basis(self, m, eigvecs):
        m = tf.expand_dims(m, axis=0)
        m_in_gen_basis = tf.matmul(self._complex(r=m), eigvecs)
        return m_in_gen_basis[0]

    def _retrieve_from_generator_basis(self, m_in_gen_basis, eigvecs):
        m = tf.matmul(m_in_gen_basis, tf.linalg.adjoint(eigvecs))
        return tf.math.real(m)

    def _l2_normalize(self, x, axis=None):
        if axis is None:
            norm = tf.sqrt(tf.reduce_sum(x * tf.math.conj(x), keepdims=True))
        else:
            norm = tf.sqrt(tf.reduce_sum(x * tf.math.conj(x), axis=axis, keepdims=True))
        return x / (norm + self.eps)

    def _skew_symmetrize(self, a):
        a_tr = tf.transpose(a, perm=[1, 0])
        return 0.5 * (a - a_tr)

    def _complex(self, r=None, i=None):
        if r is None and i is None:
            return tf.complex(0.0, 0.0)
        if r is None:
            r = tf.zeros_like(i)
        if i is None:
            i = tf.zeros_like(r)
        return tf.complex(tf.cast(r, tf.float32), tf.cast(i, tf.float32))

    def _create_resolution_filters(self):
        if self.use_zero_padded_filter:
            d_out = tf.reduce_prod(tf.convert_to_tensor(self.axes_widths, dtype=tf.int32))
            self.resolution_filters = tf.Variable(
                tf.zeros([self.num_resolution_filters, d_out]),
                trainable=True,
                name="resolution_filters",
                dtype=tf.float32,
            )
        else:
            self.resolution_filters = tf.Variable(
                tf.zeros([self.num_resolution_filters, self.n_input_dims]),
                trainable=True,
                name="resolution_filters",
                dtype=tf.float32,
            )

    def _create_lifting_offsets(self):
        self.lifting_offsets = []
        for i in range(self.group_dims):
            self.lifting_offsets.append(tf.Variable(
                tf.zeros([1]),
                trainable=True,
                name=f"lifting_offset_{i}",
                dtype=tf.float32,
            ))

    def _create_eigenvalue_parametrizations(self):
        n = self.n_input_dims
        initial_phases = tf.random.normal([self.group_dims, n], 0, 1e-3)
        self._eigenvalue_parametrizations = tf.Variable(
            initial_phases,
            trainable=True,
            name="generator_eigenvalue_parametrization",
        )

    def _create_eigenvector_parametrization(self):
        n = self.n_input_dims
        self._eigenvector_parametrization = tf.Variable(
            tf.random.normal(shape=[n, n], stddev=1e-3),
            trainable=True,
            name="eigenvector_parametrization",
        )

    def _create_step_counter(self):
        self.step_counter = tf.Variable(
            tf.zeros([], dtype=tf.uint32), trainable=False, name="step_counter"
        )

    @property
    def generators(self):
        evecs, evals = self._decompose_generator_parametrization(self._eigenvector_parametrization)
        generators = []
        for i in range(self.group_dims):
            if self.group_dims > 1:
                evals = self._compute_eigenvalues(self._eigenvalue_parametrizations[i], evecs)
            D = tf.linalg.diag(tf.exp(evals))
            G = evecs @ D @ tf.linalg.adjoint(evecs)
            generators.append(tf.math.real(G))
        return tf.stack(generators, axis=0)

    @property
    def generator_phases(self):
        evecs, evals = self._decompose_generator_parametrization(self._eigenvector_parametrization)
        generator_phases = []
        for i in range(self.group_dims):
            if self.group_dims > 1:
                evals = self._compute_eigenvalues(self._eigenvalue_parametrizations[i], evecs)
            generator_phases.append(tf.math.imag(evals))
        return tf.stack(generator_phases, axis=0)

    @property
    def log_generators(self):
        evecs, evals = self._decompose_generator_parametrization(self._eigenvector_parametrization)
        log_generators = []
        for i in range(self.group_dims):
            if self.group_dims > 1:
                evals = self._compute_eigenvalues(self._eigenvalue_parametrizations[i], evecs)
            D = tf.linalg.diag(evals)
            log_G = evecs @ D @ tf.linalg.adjoint(evecs)
            log_generators.append(tf.math.real(log_G))
        return tf.stack(log_generators, axis=0)

    @property
    def num_lifted_channels(self):
        return self.num_resolution_filters

    @property
    def n_input_dims(self):
        return math.prod(self._input_shape[1:])

    @property
    def batchsize(self):
        return self._input_shape[0]
