import numpy as np
import tensorflow as tf
import time
import os
from modeling_utils import accuracy, calc_loss, ada_training

from readers.path_reader import distorted_inputs, inputs
test_data = './data/test.txt'
valid_data = './data/valid.txt'
train_data = './data/train.txt'

# Global variables
num_labels = 10
valid_records = 5000
test_records = 10000
train_records = 55000

# Model variables
patch_one = 1
patch_three = 3
patch_five = 5
depth_five = 4
depth1 = 8
depth2 = 8

layer1 = 1024
layer2 = 512
hidden_dprob = 0.5
IMAGE_SIZE = 24
batch_size = 100
num_channels = 1


def evaluate(test_set, path):
    """
    Function that evaluates model performance
    :param test_set: Data to use
    :param path: The bath to the current model
    :return: % accuracy against the supplied dataset
    """
    with tf.Graph().as_default():

        images, labels = inputs(test_set, num_threads=10)

        logits = inference(train=False, images=images)
        test_acc = accuracy(logits, labels)

        saver = tf.train.Saver(tf.all_variables())

        sess = tf.Session()
        coord = tf.train.Coordinator()
        saver.restore(sess=sess, save_path=path)

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        try:
            true_count = 0
            if test_set == valid_data:
                num_records = valid_records
            else:
                num_records = test_records

            step = 0
            while step < int(num_records / batch_size):
                acc = sess.run(test_acc)
                true_count += np.sum(acc)
                step += 1

        except tf.errors.OutOfRangeError as e:
            print 'Issues: ', e
        finally:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            sess.close()

        return 100 * (float(true_count) / num_records)


def inference(train, images):
    """
    Run inference on the defined model.
    :param train: Boolean to designate if this is for training
    :param images: Supplied images
    """
    # Variables.
    l1_conv_one_w1 = tf.Variable(tf.truncated_normal([patch_one, patch_one, num_channels, depth1], stddev=0.1))
    l1_conv_one_b1 = tf.Variable(tf.zeros([depth1]))

    l1_conv_three_w1 = tf.Variable(tf.truncated_normal([patch_three, patch_three, num_channels, depth1], stddev=0.1))
    l1_conv_three_b1 = tf.Variable(tf.constant(0.1, shape=[depth1]))

    l1_conv_five_w1 = tf.Variable(tf.truncated_normal([patch_five, patch_five, num_channels, depth1], stddev=0.1))
    l1_conv_five_b1 = tf.Variable(tf.constant(0.1, shape=[depth1]))

    l2_conv_one_w1 = tf.Variable(tf.truncated_normal([patch_one, patch_one, depth1 * 3 + num_channels, depth1], stddev=0.1))
    l2_conv_one_w2 = tf.Variable(tf.truncated_normal([patch_one, patch_one, depth1 * 3 + num_channels, depth1], stddev=0.1))
    l2_conv_one_w3 = tf.Variable(tf.truncated_normal([patch_one, patch_one, depth1 * 3 + num_channels, depth1], stddev=0.1))
    l2_conv_one_w4 = tf.Variable(tf.truncated_normal([patch_one, patch_one, depth1 * 3 + num_channels, depth1], stddev=0.1))
    l2_conv_one_b1 = tf.Variable(tf.zeros([depth1]))
    l2_conv_one_b2 = tf.Variable(tf.zeros([depth1]))
    l2_conv_one_b3 = tf.Variable(tf.zeros([depth1]))
    l2_conv_one_b4 = tf.Variable(tf.zeros([depth1]))

    l2_conv_three_w1 = tf.Variable(tf.truncated_normal([patch_three, patch_three, depth1, depth1], stddev=0.1))
    l2_conv_three_b1 = tf.Variable(tf.constant(0.1, shape=[depth1]))

    l2_conv_five_w1 = tf.Variable(tf.truncated_normal([patch_five, patch_five, depth1, depth1], stddev=0.1))
    l2_conv_five_b1 = tf.Variable(tf.constant(0.1, shape=[depth1]))

    l3_conv_one_w1 = tf.Variable(
        tf.truncated_normal([patch_one, patch_one, depth1 * 4, depth2], stddev=0.1))
    l3_conv_one_w2 = tf.Variable(
        tf.truncated_normal([patch_one, patch_one, depth1 * 4, depth2], stddev=0.1))
    l3_conv_one_w3 = tf.Variable(
        tf.truncated_normal([patch_one, patch_one, depth1 * 4, depth2], stddev=0.1))
    l3_conv_one_w4 = tf.Variable(
        tf.truncated_normal([patch_one, patch_one, depth1 * 4, depth2], stddev=0.1))
    l3_conv_one_b1 = tf.Variable(tf.zeros([depth2]))
    l3_conv_one_b2 = tf.Variable(tf.zeros([depth2]))
    l3_conv_one_b3 = tf.Variable(tf.zeros([depth2]))
    l3_conv_one_b4 = tf.Variable(tf.zeros([depth2]))

    l3_conv_three_w1 = tf.Variable(tf.truncated_normal([patch_three, patch_three, depth2, depth2], stddev=0.1))
    l3_conv_three_b1 = tf.Variable(tf.constant(0.1, shape=[depth2]))

    l3_conv_five_w1 = tf.Variable(tf.truncated_normal([patch_five, patch_five, depth2, depth2], stddev=0.1))
    l3_conv_five_b1 = tf.Variable(tf.constant(0.1, shape=[depth2]))

    fcw1 = tf.Variable(tf.truncated_normal([IMAGE_SIZE/(2*2) * IMAGE_SIZE/(2*2) * depth2 * 4, layer1], stddev=0.1))
    fcb1 = tf.Variable(tf.constant(0.1, shape=[layer1]))
    fcw2 = tf.Variable(tf.truncated_normal([layer1, layer2], stddev=0.1))
    fcb2 = tf.Variable(tf.constant(0.1, shape=[layer2]))
    fcw3 = tf.Variable(tf.truncated_normal([layer2, num_labels], stddev=0.1))
    fcb3 = tf.Variable(tf.constant(0.1, shape=[num_labels]))

    # Model. Both models for training and testing. Regularization techniques are taken out for testing/inference
    # purposes
    # This is an implementation of the inception architecture.
    def train_model(data):
        l1_conv_one = tf.nn.conv2d(data, l1_conv_one_w1, [1, 1, 1, 1], padding="SAME")
        l1_one_relu = tf.nn.relu(l1_conv_one + l1_conv_one_b1)
        l1_conv_three = tf.nn.conv2d(data, l1_conv_three_w1, [1, 1, 1, 1], padding="SAME")
        l1_three_relu = tf.nn.relu(l1_conv_three + l1_conv_three_b1)
        l1_conv_five = tf.nn.conv2d(data, l1_conv_five_w1, [1, 1, 1, 1], padding="SAME")
        l1_five_relu = tf.nn.relu(l1_conv_five + l1_conv_five_b1)
        l1_pool = tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")

        l1_out = tf.concat(concat_dim=3, values=[l1_one_relu, l1_three_relu, l1_five_relu, l1_pool])

        l2_conv_one = tf.nn.conv2d(l1_out, l2_conv_one_w1, [1, 1, 1, 1], padding="SAME")
        l2_one_relu = tf.nn.relu(l2_conv_one + l2_conv_one_b1)
        l2_conv_pre_three = tf.nn.conv2d(l1_out, l2_conv_one_w2, [1, 1, 1, 1], padding="SAME")
        l2_pre_three_relu = tf.nn.relu(l2_conv_pre_three + l2_conv_one_b2)
        l2_conv_three = tf.nn.conv2d(l2_pre_three_relu, l2_conv_three_w1, [1, 1, 1, 1], padding="SAME")
        l2_three_relu = tf.nn.relu(l2_conv_three + l2_conv_three_b1)
        l2_conv_pre_five = tf.nn.conv2d(l1_out, l2_conv_one_w3, [1, 1, 1, 1], padding="SAME")
        l2_pre_five_relu = tf.nn.relu(l2_conv_pre_five + l2_conv_one_b3)
        l2_conv_five = tf.nn.conv2d(l2_pre_five_relu, l2_conv_five_w1, [1, 1, 1, 1], padding="SAME")
        l2_five_relu = tf.nn.relu(l2_conv_five + l2_conv_five_b1)
        l2_pool = tf.nn.max_pool(l1_out, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")
        l2_pool_proj = tf.nn.conv2d(l2_pool, l2_conv_one_w4, [1, 1, 1, 1], padding="SAME")
        l2_pool_proj_relu = tf.nn.relu(l2_pool_proj + l2_conv_one_b4)

        l2_out = tf.concat(concat_dim=3, values=[l2_one_relu, l2_three_relu, l2_five_relu, l2_pool_proj_relu])
        l2_pool = tf.nn.max_pool(l2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        l3_conv_one = tf.nn.conv2d(l2_pool, l3_conv_one_w1, [1, 1, 1, 1], padding="SAME")
        l3_one_relu = tf.nn.relu(l3_conv_one + l3_conv_one_b1)
        l3_conv_pre_three = tf.nn.conv2d(l2_pool, l3_conv_one_w2, [1, 1, 1, 1], padding="SAME")
        l3_pre_three_relu = tf.nn.relu(l3_conv_pre_three + l3_conv_one_b2)
        l3_conv_three = tf.nn.conv2d(l3_pre_three_relu, l3_conv_three_w1, [1, 1, 1, 1], padding="SAME")
        l3_three_relu = tf.nn.relu(l3_conv_three + l3_conv_three_b1)
        l3_conv_pre_five = tf.nn.conv2d(l2_pool, l3_conv_one_w3, [1, 1, 1, 1], padding="SAME")
        l3_pre_five_relu = tf.nn.relu(l3_conv_pre_five + l3_conv_one_b3)
        l3_conv_five = tf.nn.conv2d(l3_pre_five_relu, l3_conv_five_w1, [1, 1, 1, 1], padding="SAME")
        l3_five_relu = tf.nn.relu(l3_conv_five + l3_conv_five_b1)
        l3_pool = tf.nn.max_pool(l2_pool, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")
        l3_pool_proj = tf.nn.conv2d(l3_pool, l3_conv_one_w4, [1, 1, 1, 1], padding="SAME")
        l3_pool_proj_relu = tf.nn.relu(l3_pool_proj + l3_conv_one_b4)

        l3_out = tf.concat(concat_dim=3, values=[l3_one_relu, l3_three_relu, l3_five_relu, l3_pool_proj_relu])
        l3_pool = tf.nn.max_pool(l3_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        shape = l3_pool.get_shape().as_list()
        reshape = tf.reshape(l3_pool, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, fcw1) + fcb1)
        hidden_dropout = tf.nn.dropout(hidden, hidden_dprob)
        hidden2 = tf.nn.relu(tf.matmul(hidden_dropout, fcw2) + fcb2)
        hidden2_dropout = tf.nn.dropout(hidden2, hidden_dprob)
        output = tf.matmul(hidden2_dropout, fcw3) + fcb3
        return output

    def test_model(data):
        l1_conv_one = tf.nn.conv2d(data, l1_conv_one_w1, [1, 1, 1, 1], padding="SAME")
        l1_one_relu = tf.nn.relu(l1_conv_one + l1_conv_one_b1)
        l1_conv_three = tf.nn.conv2d(data, l1_conv_three_w1, [1, 1, 1, 1], padding="SAME")
        l1_three_relu = tf.nn.relu(l1_conv_three + l1_conv_three_b1)
        l1_conv_five = tf.nn.conv2d(data, l1_conv_five_w1, [1, 1, 1, 1], padding="SAME")
        l1_five_relu = tf.nn.relu(l1_conv_five + l1_conv_five_b1)
        l1_pool = tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")

        l1_out = tf.concat(concat_dim=3, values=[l1_one_relu, l1_three_relu, l1_five_relu, l1_pool])

        l2_conv_one = tf.nn.conv2d(l1_out, l2_conv_one_w1, [1, 1, 1, 1], padding="SAME")
        l2_one_relu = tf.nn.relu(l2_conv_one + l2_conv_one_b1)
        l2_conv_pre_three = tf.nn.conv2d(l1_out, l2_conv_one_w2, [1, 1, 1, 1], padding="SAME")
        l2_pre_three_relu = tf.nn.relu(l2_conv_pre_three + l2_conv_one_b2)
        l2_conv_three = tf.nn.conv2d(l2_pre_three_relu, l2_conv_three_w1, [1, 1, 1, 1], padding="SAME")
        l2_three_relu = tf.nn.relu(l2_conv_three + l2_conv_three_b1)
        l2_conv_pre_five = tf.nn.conv2d(l1_out, l2_conv_one_w3, [1, 1, 1, 1], padding="SAME")
        l2_pre_five_relu = tf.nn.relu(l2_conv_pre_five + l2_conv_one_b3)
        l2_conv_five = tf.nn.conv2d(l2_pre_five_relu, l2_conv_five_w1, [1, 1, 1, 1], padding="SAME")
        l2_five_relu = tf.nn.relu(l2_conv_five + l2_conv_five_b1)
        l2_pool = tf.nn.max_pool(l1_out, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")
        l2_pool_proj = tf.nn.conv2d(l2_pool, l2_conv_one_w4, [1, 1, 1, 1], padding="SAME")
        l2_pool_proj_relu = tf.nn.relu(l2_pool_proj + l2_conv_one_b4)

        l2_out = tf.concat(concat_dim=3, values=[l2_one_relu, l2_three_relu, l2_five_relu, l2_pool_proj_relu])
        l2_pool = tf.nn.max_pool(l2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        l3_conv_one = tf.nn.conv2d(l2_pool, l3_conv_one_w1, [1, 1, 1, 1], padding="SAME")
        l3_one_relu = tf.nn.relu(l3_conv_one + l3_conv_one_b1)
        l3_conv_pre_three = tf.nn.conv2d(l2_pool, l3_conv_one_w2, [1, 1, 1, 1], padding="SAME")
        l3_pre_three_relu = tf.nn.relu(l3_conv_pre_three + l3_conv_one_b2)
        l3_conv_three = tf.nn.conv2d(l3_pre_three_relu, l3_conv_three_w1, [1, 1, 1, 1], padding="SAME")
        l3_three_relu = tf.nn.relu(l3_conv_three + l3_conv_three_b1)
        l3_conv_pre_five = tf.nn.conv2d(l2_pool, l3_conv_one_w3, [1, 1, 1, 1], padding="SAME")
        l3_pre_five_relu = tf.nn.relu(l3_conv_pre_five + l3_conv_one_b3)
        l3_conv_five = tf.nn.conv2d(l3_pre_five_relu, l3_conv_five_w1, [1, 1, 1, 1], padding="SAME")
        l3_five_relu = tf.nn.relu(l3_conv_five + l3_conv_five_b1)
        l3_pool = tf.nn.max_pool(l2_pool, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")
        l3_pool_proj = tf.nn.conv2d(l3_pool, l3_conv_one_w4, [1, 1, 1, 1], padding="SAME")
        l3_pool_proj_relu = tf.nn.relu(l3_pool_proj + l3_conv_one_b4)

        l3_out = tf.concat(concat_dim=3, values=[l3_one_relu, l3_three_relu, l3_five_relu, l3_pool_proj_relu])
        l3_pool = tf.nn.max_pool(l3_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        shape = l3_pool.get_shape().as_list()
        reshape = tf.reshape(l3_pool, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, fcw1) + fcb1)
        hidden2 = tf.nn.relu(tf.matmul(hidden, fcw2) + fcb2)
        output = tf.matmul(hidden2, fcw3) + fcb3
        return output

    if train:
        logits = train_model(images)
    else:
        logits = test_model(images)

    return logits


def run_training(lr, num_epochs, path):
    """
    Run the training for the model
    :param lr: The learning rate
    :param num_epochs: The number of full passes through the training data
    :param path: Path to save/restore the model
    """
    with tf.Graph().as_default():

        # Pull in the images using the defined readers and perform distortions for training
        train_images, train_labels = distorted_inputs(train_data, num_threads=4)

        # Get predictions and variables to save for semantic segmentation
        logits = inference(train=True, images=train_images)

        # Calculate the loss between predictions and labels
        loss = calc_loss(logits, train_labels)

        # Perform Adam optimization
        train_op = ada_training(loss, learning_rate=lr)

        # Initialize saver, variables, and session
        saver = tf.train.Saver(tf.all_variables())
        init_op = tf.initialize_all_variables()
        sess = tf.Session()

        # Restore or initialize the model parameters
        if os.path.isfile(path):
            saver.restore(sess=sess, save_path=path)
            print 'Model Restored'
        else:
            sess.run(init_op)
            print 'Model Initialized'

        tf.train.start_queue_runners(sess=sess)

        # Output information as the network trains. Every half epoch output various statistics and
        # Every full epoch, output validation error.
        for step in xrange(int((num_epochs * train_records) / batch_size)):

            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            if step % 275 == 0 or step == int((num_epochs * train_records) / batch_size):
                print "------------------------------------------"
                print "Examples/sec: ", batch_size / duration
                print "Sec/batch: ", float(duration)
                print "Current epoch: ", (float(step) * batch_size) / train_records
                print "Current learning rate: ", lr
                print "Minibatch loss at step", step, ":", loss_value
            if step % 550 == 0 or step == int((num_epochs * train_records) / batch_size) - 1:
                save_path = saver.save(sess, path)
                print "Model saved in file: ", save_path
                print "Validation accuracy: ", evaluate(valid_data, path)

        print "===================================="
        print "Test accuracy: ", evaluate(test_data, path)


def inception_model(lr, num_epochs):
    """
    Run the inception model
    :param lr: The learning rate
    :param num_epochs: The number of full passes through the training data
    :return:
    """
    checkpoint_file = './incept/incept.ckpt'
    wd = os.getcwd()
    if not os.path.exists(wd + '/incept'):
        os.mkdir(wd + '/incept')

    run_training(lr, num_epochs, checkpoint_file)