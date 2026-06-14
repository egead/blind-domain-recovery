import tensorflow as tf
import numpy as np
import core.regularizations as ri
import time
import datetime

def get_ordered_regularizations_coeffs(
    uniformity_maximization_reg_coeff=0.0,
    total_correlation_minimization_reg_coeff=0.0,
    joint_entropy_maximization_reg_coeff=0.0,
    uniformity_estimator_reg_coeff=0.0,
    probability_estimator_reg_coeff=0.0
):
    reg_coeffs = [
        [
            ri.UNIFORMITY_MAXIMIZATION_REGULARIZATION_ORDER,
            uniformity_maximization_reg_coeff,
        ],
        [
            ri.TOTAL_CORRELATION_MINIMIZATION_REGULARIZATION_ORDER,
            total_correlation_minimization_reg_coeff,
        ],
        [
            ri.JOINT_ENTROPY_MAXIMIZATION_REGULARIZATION_ORDER,
            joint_entropy_maximization_reg_coeff,
        ],
        [
            ri.UNIFORMITY_ESTIMATOR_REGULARIZATION_ORDER,
            uniformity_estimator_reg_coeff
        ],
        [
            ri.PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER,
            probability_estimator_reg_coeff
            
        ]
    ]

    regularization_coeffs = tf.convert_to_tensor(reg_coeffs, dtype=tf.float32)
    sorting_idxs = tf.argsort(regularization_coeffs[:, 0], axis=0)
    regularization_coeffs = tf.gather(regularization_coeffs[:, 1], sorting_idxs)

    return regularization_coeffs


def get_ordered_regularization_terms(model):
    regularization_tensor = tf.convert_to_tensor(model.losses)
    orders = tf.cast(regularization_tensor[:, 0], tf.int32)
    values = regularization_tensor[:, 1]
    num_reg_types = 5
    return tf.math.unsorted_segment_sum(values, orders, num_segments=num_reg_types)


def get_loss(regularizations, **kwargs):
    reg_coeffs = get_ordered_regularizations_coeffs(**kwargs)
    reg_coeffs = reg_coeffs[0:len(regularizations)]
    
    return tf.reduce_sum(regularizations * reg_coeffs)


def split_weights(weight_list, included_modules):
    included_weights = []
    excluded_weights = []
    for weight in weight_list:
        name_list = weight.name.split("/")
        included = False

        for module in included_modules:
            for name in name_list:
                if module in name[0:len(module)]:
                    included = True
                
            if included:
                break

        if included:
            included_weights.append(weight)
        else:
            excluded_weights.append(weight)

    return included_weights, excluded_weights

def compute_total_lr_scaled_training_steps(num_epochs, num_steps, lr_scheduler):
    lr_scaled_training_steps = 0.0

    for epoch in range(num_epochs):
        lr = lr_scheduler.get_learning_rate(epoch)
        lr_scaled_training_steps = lr_scaled_training_steps + lr * num_steps

    return lr_scaled_training_steps

def log(epoch, step, lr_scaled_normalized_training_time, loss, regularizations):
    loss_term_ids = [
        ri.UNIFORMITY_MAXIMIZATION_REGULARIZATION_ORDER,
        ri.TOTAL_CORRELATION_MINIMIZATION_REGULARIZATION_ORDER,
        ri.JOINT_ENTROPY_MAXIMIZATION_REGULARIZATION_ORDER,
        ri.UNIFORMITY_ESTIMATOR_REGULARIZATION_ORDER,
        ri.PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER
    ]
    
    loss_term_descriptions = [
        "uniformity",
        "total_correlation",
        "h_joint",
        "uniformity_estimator",
        "probability_estimator"
    ]

    regularizations = regularizations.numpy()
    loss_term_ids = np.array(loss_term_ids)
    sorting_idxs = list(np.argsort(loss_term_ids))

    msg = "Epoch: " + str(epoch) + " Step: " + str(step) + ", " 
    msg = msg + "Normalized time:{:.2f}".format(lr_scaled_normalized_training_time) + ", "
    msg = msg + "Loss:{:.2f}".format(loss) + ", "

    for i in range(len(regularizations)):
        msg = msg + loss_term_descriptions[sorting_idxs[i]] + ":"
        msg = msg + "{:.2f}".format(regularizations[i]) + ", "

    msg = msg[:-2]
    msg = msg + "\r"

    print(msg)
    
def run_train_step_eager(
    model,
    optimizer_model,
    optimizer_estimator,
    lr_scaled_normalized_training_time,
    x_batch,
    model_loss_coeffs,
    estimator_loss_coeffs,
    model_weights,
    estimator_weights
):
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        outputs = model(x_batch, lr_scaled_normalized_training_time, training=True)
    
        # Compute regularizations
        regularizations = get_ordered_regularization_terms(model)
    
        # Compute losses
        model_loss = get_loss(regularizations, **model_loss_coeffs)
        estimator_loss = get_loss(regularizations, **estimator_loss_coeffs)

    # Compute gradients independently
    grads_model = tape.gradient(model_loss, model_weights)
    grads_estimator = tape.gradient(estimator_loss, estimator_weights)

    # Apply updates
    optimizer_model.apply_gradients(zip(grads_model, model_weights))
    optimizer_estimator.apply_gradients(zip(grads_estimator, estimator_weights))

    return model_loss, regularizations

# ------------------------------
# Replica-local training step
# ------------------------------

@tf.function
def run_train_step(
    model,
    optimizer_model,
    optimizer_estimator,
    lr_scaled_normalized_training_time,
    x_batch,
    model_loss_coeffs,
    estimator_loss_coeffs,
    model_weights,
    estimator_weights
):
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        outputs = model(x_batch, lr_scaled_normalized_training_time, training=True)
    
        # Compute regularizations
        regularizations = get_ordered_regularization_terms(model)
    
        # Compute losses
        model_loss = get_loss(regularizations, **model_loss_coeffs)
        estimator_loss = get_loss(regularizations, **estimator_loss_coeffs)

    # Compute gradients independently
    grads_model = tape.gradient(model_loss, model_weights)
    grads_estimator = tape.gradient(estimator_loss, estimator_weights)

    # Apply updates
    optimizer_model.apply_gradients(zip(grads_model, model_weights))
    optimizer_estimator.apply_gradients(zip(grads_estimator, estimator_weights))

    return model_loss, regularizations


@tf.function
def distributed_train_step(
    strategy,
    model,
    optimizer_model,
    optimizer_estimator,
    lr_scaled_normalized_training_time,
    x_batch,
    model_loss_coeffs,
    estimator_loss_coeffs,
    model_weights,
    estimator_weights
):
    """
    Runs a single distributed training step safely with multi-GPU.
    - model_weights and estimator_weights are precomputed outside the graph.
    """

    def step_fn(inputs):
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            outputs = model(inputs, lr_scaled_normalized_training_time, training=True)
    
            # Compute regularizations
            regularizations = get_ordered_regularization_terms(model)
    
            # Compute losses
            model_loss = get_loss(regularizations, **model_loss_coeffs)
            estimator_loss = get_loss(regularizations, **estimator_loss_coeffs)

        # Compute gradients independently
        grads_model = tape.gradient(model_loss, model_weights)
        grads_estimator = tape.gradient(estimator_loss, estimator_weights)

        # Apply updates
        optimizer_model.apply_gradients(zip(grads_model, model_weights))
        optimizer_estimator.apply_gradients(zip(grads_estimator, estimator_weights))

        return model_loss, regularizations

    per_replica_loss, per_replica_regs = strategy.run(step_fn, args=(x_batch,))
    # Reduce mean across replicas
    loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
    regularizations = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_regs, axis=None)

    return loss, regularizations


def fit(
    model,
    optimizer_model,
    optimizer_estimators,
    lr_scheduler_model,
    lr_scheduler_estimators,
    dataset,
    batch_size,
    epochs,
    early_stop_epoch=None,
    callbacks=None,
    model_loss_coeffs=None,
    estimator_loss_coeffs=None,
    strategy=None,
    eager=False,
    shuffle_buffer=20000
):
    """
    Multi-GPU / single-GPU training loop using the safe distributed step.
    """
    steps_per_epoch = tf.data.experimental.cardinality(dataset).numpy() // batch_size
     
    total_lr_scaled_training_steps = compute_total_lr_scaled_training_steps(
        epochs, steps_per_epoch, lr_scheduler_model
    )
    lr_scaled_training_steps = 0.0

    # Precompute model and estimator weights outside the graph
    estimator_weights, model_weights = split_weights(
        model.trainable_weights,
        ["uniformity_estimator", "probability_estimator"]
    )

    print("Number of elements in dataset:", tf.data.experimental.cardinality(dataset).numpy())

    start_time = None
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")

        # Shuffle + batch dataset fresh each epoch
        ds_epoch = dataset.shuffle(shuffle_buffer, seed=epoch, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)

        # Distribute if strategy provided
        dist_dataset = strategy.experimental_distribute_dataset(ds_epoch) if strategy else ds_epoch

        # Update learning rates
        lr_model = lr_scheduler_model.get_learning_rate(epoch)
        lr_estimators = lr_scheduler_estimators.get_learning_rate(epoch)
        
        tf.keras.backend.set_value(optimizer_model.lr, lr_model)
        tf.keras.backend.set_value(optimizer_estimators.lr, lr_estimators)

        for step, x_batch in enumerate(dist_dataset):
            lr_scaled_training_steps += lr_model
            lr_scaled_normalized_training_time = lr_scaled_training_steps / total_lr_scaled_training_steps

            if strategy and not eager:
                # Multi-GPU safe distributed step
                loss, regularizations = distributed_train_step(
                    strategy,
                    model,
                    optimizer_model,
                    optimizer_estimators,
                    lr_scaled_normalized_training_time,
                    x_batch,
                    model_loss_coeffs,
                    estimator_loss_coeffs,
                    model_weights,
                    estimator_weights
                )
            elif not eager:
                # Single-GPU
                loss, regularizations = run_train_step(
                    model,
                    optimizer_model,
                    optimizer_estimators,
                    lr_scaled_normalized_training_time,
                    x_batch,
                    model_loss_coeffs,
                    estimator_loss_coeffs,
                    model_weights,
                    estimator_weights
                )
            else:
                # Eager mode
                loss, regularizations = run_train_step_eager(
                    model,
                    optimizer_model,
                    optimizer_estimators,
                    lr_scaled_normalized_training_time,
                    x_batch,
                    model_loss_coeffs,
                    estimator_loss_coeffs,
                    model_weights,
                    estimator_weights
                )
                
            if step % 20 == 0:
                log(epoch, step, lr_scaled_normalized_training_time, loss, regularizations)

        # Callbacks at epoch end
        if callbacks:
            for cb in callbacks:
                cb.on_epoch_end(epoch)

        if ((early_stop_epoch is not None) and (epoch >= early_stop_epoch)):
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            if start_time is not None:
                elapsed_time = time.time() - start_time    
                eta_seconds = elapsed_time * (epochs - epoch) / 10
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(f"Remaining time:{eta_str}")
            
            start_time = time.time()
    
        

            




