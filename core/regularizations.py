import tensorflow as tf

UNIFORMITY_MAXIMIZATION_REGULARIZATION_ORDER = 0
TOTAL_CORRELATION_MINIMIZATION_REGULARIZATION_ORDER = 1
JOINT_ENTROPY_MAXIMIZATION_REGULARIZATION_ORDER = 2
UNIFORMITY_ESTIMATOR_REGULARIZATION_ORDER = 3
PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER = 4

REGULARIZATIONS = {
    "uniformity_maximization": {"order": UNIFORMITY_MAXIMIZATION_REGULARIZATION_ORDER},
    "total_correlation_minimization": {
        "order": TOTAL_CORRELATION_MINIMIZATION_REGULARIZATION_ORDER
    },
    "joint_entropy_maximization": {"order": JOINT_ENTROPY_MAXIMIZATION_REGULARIZATION_ORDER},
    "uniformity_estimator_regularization": {
        "order": UNIFORMITY_ESTIMATOR_REGULARIZATION_ORDER
    },
    "probability_estimator_regularization": {
        "order": PROBABILITY_ESTIMATOR_REGULARIZATION_ORDER
    }
}

def convert_to_regularization_format(regularization_key, value):
    order = REGULARIZATIONS[regularization_key]["order"]
    return tf.convert_to_tensor([order, value], dtype=tf.float32)