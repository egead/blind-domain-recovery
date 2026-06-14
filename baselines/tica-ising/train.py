import tensorflow as tf
import numpy as np
import time
import os

# Import the pruned generator and the updated Ising TICA model
from synthetic_data_generator import DataGenerator
from tica import TICAModel1D

def compute_zca_matrix(data, epsilon=1e-5):
    """Computes the ZCA Whitening matrix for a given dataset."""
    mean = tf.reduce_mean(data, axis=0)
    centered = data - mean
    
    # Calculate the covariance matrix
    cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(centered)[0], tf.float32)
    
    # Eigen decomposition
    e, v = tf.linalg.eigh(cov)
    
    # ZCA = V * (S + eps)^(-0.5) * V^T
    zca_matrix = tf.matmul(v, tf.matmul(tf.linalg.diag(tf.math.rsqrt(e + epsilon)), v, transpose_b=True))
    
    return mean, zca_matrix

def run_tica_experiment_ising(exp_name, exp_specs, save_dir=None):
    # --- 1. Extract Hyperparameters ---
    BATCH_SIZE = exp_specs.get("batch_size", 250)
    N_COMPONENTS = exp_specs.get("n_components", 33) # 33 for Ising
    POOL_SIZE = exp_specs.get("pool_size", 5)
    USE_ORTHOGONAL = exp_specs.get("use_orthogonal", True)
    IS_CIRCULANT = exp_specs.get("is_circulant", False) # False for Ising (open chain)
    LEARNING_RATE = exp_specs.get("learning_rate", 0.01)
    EPOCHS = exp_specs.get("epochs", 1500)
    STEPS_PER_EPOCH = exp_specs.get("steps_per_epoch", 1000)
    OUTPUT_REP = exp_specs.get("output_representation", "natural") 
    
    INPUT_DIM = N_COMPONENTS
    TOTAL_SAMPLES = STEPS_PER_EPOCH * BATCH_SIZE

    # Ensure save directory exists early on so we can save ZCA matrices
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 2. Pre-generate the ENTIRE Dataset into RAM ---
    print(f"Generating {TOTAL_SAMPLES} Ising samples into memory for {exp_name}...")
    
    # Setup for Ising Model 
    generator = DataGenerator(
        batch_size=BATCH_SIZE,
        features=[{"type": "ising", 
                   "beta_min": 1.0, 
                   "beta_max": 5.0, 
                   "n_gibbs_steps": 10}],
        n_components=N_COMPONENTS,
        is_circulant=IS_CIRCULANT,
        output_representation=OUTPUT_REP
    )

    batches = []
    for i in range(STEPS_PER_EPOCH):
        batch = generator.sample_batch_of_data()
        batches.append(batch)
        if (i + 1) % 100 == 0:
            print(f"Generated batch: {i + 1}/{STEPS_PER_EPOCH}")
    
    # Ising spins are already +/- 1, no need for pixel normalization
    full_dataset_np = np.concatenate(batches, axis=0).astype(np.float32)
    
    # --- 3. Pre-compute and Apply ZCA to the entire dataset ---
    print("Computing and applying ZCA Whitening matrix...")
    full_dataset_tf = tf.convert_to_tensor(full_dataset_np)
    global_mean, zca_matrix = compute_zca_matrix(full_dataset_tf)
    
    # Save ZCA stats for validation
    if save_dir:
        np.save(os.path.join(save_dir, "zca_mean.npy"), global_mean.numpy())
        np.save(os.path.join(save_dir, "zca_matrix.npy"), zca_matrix.numpy())
        print("ZCA matrix and mean saved to disk.")
    
    centered_data = full_dataset_tf - global_mean
    whitened_data = tf.matmul(centered_data, zca_matrix)

    # --- 4. Create the Fast tf.data.Dataset Pipeline ---
    print("Building optimized tf.data pipeline...")
    dataset = tf.data.Dataset.from_tensor_slices(whitened_data)

    # --- 5. Model Setup & Unified Train Step ---
    model = TICAModel1D(
        input_dim=INPUT_DIM, 
        n_components=N_COMPONENTS,
        pool_size=POOL_SIZE, 
        use_orthogonal=USE_ORTHOGONAL,
        is_circulant=IS_CIRCULANT
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

    @tf.function
    def train_step(x_batch):
        with tf.GradientTape() as tape:
            _ = model(x_batch, training=True)
            loss = tf.reduce_sum(model.losses)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # --- 6. Execution Loop ---
    print(f"Starting 1D TICA training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        start_time = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()
        
        # Shuffle and batch dynamically at the start of each epoch
        ds_epoch = dataset.shuffle(buffer_size=20000, seed=epoch, reshuffle_each_iteration=True)
        ds_epoch = ds_epoch.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        for step, x_batch in enumerate(ds_epoch):
            loss = train_step(x_batch)
            epoch_loss_avg.update_state(loss)
            
            # Print occasionally so the log file updates
            if (step + 1) % 100 == 0:
                print(f"  -> Epoch {epoch + 1} | Step {step + 1}/{STEPS_PER_EPOCH} | Current Loss: {loss:.4f}")

        end_time = time.time()
        print(f"Epoch {epoch + 1:04d}/{EPOCHS} "
              f"| Loss: {epoch_loss_avg.result():.4f} "
              f"| Time: {end_time - start_time:.2f}s")
        
        # Save weights (on the first epoch to verify, then every 10 epochs)
        if save_dir and (epoch == 0 or (epoch + 1) % 10 == 0):
            weights_path = os.path.join(save_dir, f"tica_weights_epoch_{epoch+1}.npy")
            np.save(weights_path, model.W.numpy())

    # Final save to ensure the last epoch is captured
    if save_dir:
        final_weights_path = os.path.join(save_dir, f"tica_weights_final.npy")
        np.save(final_weights_path, model.W.numpy())

    print(f"\nTraining complete for {exp_name}.")