import tensorflow as tf
import numpy as np
import time
import os
from synthetic_data_generator import DataGenerator2d
from tica import TICAModel

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

def run_tica_experiment(exp_name, exp_specs, save_dir=None):
    # --- 1. Extract Hyperparameters ---
    BATCH_SIZE = exp_specs.get("batch_size", 250)
    GRID_X_SIZE = exp_specs.get("grid_x_size", 15)
    GRID_Y_SIZE = exp_specs.get("grid_y_size", 15)
    POOL_SIZE = exp_specs.get("pool_size", 5)
    USE_ORTHOGONAL = exp_specs.get("use_orthogonal", True)
    LEARNING_RATE = exp_specs.get("learning_rate", 0.01)
    EPOCHS = exp_specs.get("epochs", 1500)
    STEPS_PER_EPOCH = exp_specs.get("steps_per_epoch", 1000)
    
    INPUT_DIM = GRID_X_SIZE * GRID_Y_SIZE
    TOTAL_SAMPLES = STEPS_PER_EPOCH * BATCH_SIZE

    # Ensure save directory exists early on so we can save ZCA matrices
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 2. Pre-generate the ENTIRE Dataset into RAM ---
    print(f"Generating {TOTAL_SAMPLES} samples into memory for {exp_name}...")
    generator = DataGenerator2d(
        batch_size=BATCH_SIZE,
        features=[{"type": "mnist"}],
        feature_type="mnist",
        latent_x_dims=GRID_Y_SIZE,
        latent_y_dims=GRID_Y_SIZE,
        output_representation="permuted",
        flatten_output=True
    )

    batches = []
    for i in range(STEPS_PER_EPOCH):
        batches.append(generator.sample_batch_of_data())
        if (i + 1) % 100 == 0:
            print(f"Generated batch: {i + 1}/{STEPS_PER_EPOCH}")
    
    full_dataset_np = np.concatenate(batches, axis=0)
    full_dataset_np = full_dataset_np.astype(np.float32) / 255.0
    
    # --- 3. Pre-compute and Apply ZCA to the entire dataset ---
    print("Computing and applying ZCA Whitening matrix...")
    full_dataset_tf = tf.convert_to_tensor(full_dataset_np)
    global_mean, zca_matrix = compute_zca_matrix(full_dataset_tf)
    
    # CRITICAL FIX: Save ZCA stats for validation
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
    model = TICAModel(
        input_dim=INPUT_DIM, 
        grid_x_size=GRID_X_SIZE,
        grid_y_size=GRID_Y_SIZE, 
        pool_size=POOL_SIZE, 
        use_orthogonal=USE_ORTHOGONAL
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
    print(f"Starting TICA training for {EPOCHS} epochs...")

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