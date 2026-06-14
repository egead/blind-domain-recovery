import numpy as np
from os.path import join
import tensorflow as tf
import os

from core.models import Model
from core.synthetic_data_generator import DataGenerator, DataGenerator2d
from core.training_loop import fit

class LogarithmicLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, final_lr, epochs):
        super(LogarithmicLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.epochs = epochs
        self.lrs = self.calculate_lrs()

    def calculate_lrs(self):
        return np.logspace(
            np.log10(self.initial_lr), np.log10(self.final_lr), self.epochs
        )

    def get_learning_rate(self, epoch, logs=None):
        return self.lrs[epoch]


class ModelSavingCallbacks:
    def __init__(self, model, directory):
        self._model = model
        self._directory = directory
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            path = join(self._directory, "ep{}.h5".format(epoch+1))
            self._model.save_weights(path)

def create_optimizer(lr=1e-4, global_clipnorm=None):
    optimizer = tf.optimizers.Adam(lr, global_clipnorm=global_clipnorm)
    return optimizer


def create_lr_scheduler(initial_lr, final_lr, epochs):
    lr_scheduler = LogarithmicLearningRateScheduler(initial_lr, final_lr, epochs)
    return lr_scheduler


def create_model(**kwargs):
    model = Model(**kwargs)
    return model

def make_data_generator(
    seed=0,
    dims=1,
    **kwargs
):
    if dims == 1:
        dg = DataGenerator(seed=seed, **kwargs)
    elif dims == 2:
        dg = DataGenerator2d(seed=seed, **kwargs)

    return dg

def mix_stochastically(x, mixing_steps, mixing_probability, seed=0):
    rng = np.random.default_rng(seed)
    N = x.shape[0]
    x_mixed = np.zeros_like(x)
    
    for s in range(mixing_steps):
        idx = rng.permutation(N)
        x_shuffled = x[idx]
        
        # 2. Generate mask
        tmp = rng.random(size=N)
        mask = np.where(tmp < mixing_probability, 1.0, 0.0)
        mask = mask[:, None] 
        
        x_mixed += x_shuffled * mask
    
    return x_mixed / (mixing_steps * mixing_probability)

def train(
    model,
    dataset_np,
    exp_specs,
    load_path=None,
    strategy=None,
    saving_dir=".",
    eager=False,
    **kwargs
):
    """
    Train the model using a tf.data.Dataset.
    Handles multi-GPU if strategy is provided.
    """    
    epochs=exp_specs["training_duration_in_epochs"]
    model_loss_coeffs=exp_specs["model_loss_coeffs"]
    batch_size=exp_specs["data_generator_params"]["batch_size"]
    seed=exp_specs.get("seed", 0)
    estimator_loss_coeffs=exp_specs.get("estimator_loss_coeffs", 
                                        {"uniformity_estimator_reg_coeff": 1.0,
                                         "probability_estimator_reg_coeff": 1.0})
    model_optimizer_starting_lr=exp_specs.get("model_optimizer_starting_lr", 1e-4)
    model_optimizer_ending_lr=exp_specs.get("model_optimizer_ending_lr", 1e-5)
    estimators_optimizer_starting_lr=exp_specs.get("estimators_optimizer_starting_lr", 1e-3)
    estimators_optimizer_ending_lr=exp_specs.get("estimators_optimizer_ending_lr", 1e-4)
    early_stop_epoch=exp_specs.get("early_stop_epoch", None)
    apply_stochastic_mixing=exp_specs.get("apply_stochastic_mixing", False)
    stochastic_mixing_steps=exp_specs.get("stochastic_mixing_steps", 5)
    stochastic_mixing_probability=exp_specs.get("stochastic_mixing_probability", 0.5)
    model_global_gradclip=exp_specs.get("model_global_gradclip", None)
    estimator_global_gradclip=exp_specs.get("estimator_global_gradclip", None)
    
    if strategy is None:
        strategy = tf.distribute.MirroredStrategy()

    if apply_stochastic_mixing:
        dataset_np = mix_stochastically(dataset_np,
                                        seed=seed,
                                        mixing_probability=stochastic_mixing_probability, 
                                        mixing_steps=stochastic_mixing_steps)
        
    dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
    
    with strategy.scope():
        # Initialize model under scope
        model.compile()  # Required for some Keras internal states
        # Perform a single forward pass to build variables
        dataset_batched = dataset.batch(batch_size, drop_remainder=True)
        sample_batch = next(iter(dataset_batched.take(1)))
        if isinstance(sample_batch, tuple):
            sample_input = sample_batch[0]
        else:
            sample_input = sample_batch
        model(sample_input, training=True)
        model.summary()

        if load_path is not None:
            model.load_weights(load_path)
            
        # Create optimizers & LR schedulers under scope
        optimizer_model = create_optimizer(model_optimizer_starting_lr, global_clipnorm=model_global_gradclip)
        optimizer_estimators = create_optimizer(estimators_optimizer_starting_lr, global_clipnorm=estimator_global_gradclip)

        lr_scheduler_model = create_lr_scheduler(
            model_optimizer_starting_lr, model_optimizer_ending_lr, epochs
        )
        lr_scheduler_estimators = create_lr_scheduler(
            estimators_optimizer_starting_lr, estimators_optimizer_ending_lr, epochs
        )

    # Training loop
    fit(
        model=model,
        optimizer_model=optimizer_model,
        optimizer_estimators=optimizer_estimators,
        lr_scheduler_model=lr_scheduler_model,
        lr_scheduler_estimators=lr_scheduler_estimators,
        early_stop_epoch=early_stop_epoch,
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[ModelSavingCallbacks(model, saving_dir)],
        model_loss_coeffs=model_loss_coeffs,
        estimator_loss_coeffs=estimator_loss_coeffs,
        eager=eager,
        strategy=strategy,
    )
