import tensorflow as tf


def accuracy(predictions, labels):
    """
    Function to calculate the number of matches between predictions and labels
    :param predictions: The predicted labels
    :param labels: The actual labels
    """
    labels = tf.cast(labels, tf.int32)
    matches = tf.nn.in_top_k(predictions=predictions, targets=tf.arg_max(labels, 1), k=1)
    return matches


def calc_loss(logits, labels):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    return loss


def training(loss, learning_rate):
    # Stochastic Gradient Descent Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimizer


def ada_training(loss, learning_rate):
    # Adam Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return optimizer
