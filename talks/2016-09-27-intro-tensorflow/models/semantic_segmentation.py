import numpy as np
import tensorflow as tf
import time
import os
from modeling_utils import accuracy, calc_loss, training

from readers.tfrecord_reader import distorted_inputs, inputs
test_data = './data/test.tfrecords'
valid_data = './data/valid.tfrecords'
train_data = './data/train.tfrecords'

# Global variables
num_labels = 10
valid_records = 5000
test_records = 10000
train_records = 55000

# Model variables
patch_size = 5
depth1 = 16
depth2 = 32
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

        logits, _ = inference(train=False, images=images)
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


def inference(train, images, upsample=False, img_height=0, img_width=0):
    """
    Run inference on the defined model.
    :param train: Boolean to designate if this is for training
    :param images: Supplied images
    """
    # Variables.
    conv1_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth1], stddev=0.1),
                     name='conv1_w')
    conv1_b = tf.Variable(tf.zeros([depth1]), name='conv1_b')

    conv2_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth1, depth1], stddev=0.1), name='conv2_w')
    conv2_b = tf.Variable(tf.constant(0.1, shape=[depth1]), name='conv2_b')

    conv3_w = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth1, depth2], stddev=0.1), name='conv3_w')
    conv3_b = tf.Variable(tf.constant(0.1, shape=[depth2]), name='conv3_b')

    fc1_w = tf.Variable(tf.truncated_normal([IMAGE_SIZE / (2*2*2), IMAGE_SIZE / (2*2*2), depth2, layer1], stddev=0.1),
                        name='fc1_w')
    fc1_b = tf.Variable(tf.constant(0.1, shape=[layer1]), name='fc1_b')
    fc2_w = tf.Variable(tf.truncated_normal([1, 1, layer1, layer2], stddev=0.1), name='fc2_w')
    fc2_b = tf.Variable(tf.constant(0.1, shape=[layer2]), name='fc2_b')
    fc3_w = tf.Variable(tf.truncated_normal([1, 1, layer2, num_labels], stddev=0.1), name='fc3_w')
    fc3_b = tf.Variable(tf.constant(0.1, shape=[num_labels]), name='fc3_b')

    # Model.
    def train_model(data):
        # 3 basic convolution and pooling layers followed by 3 fully convolutional ILO fully connected layers
        conv1 = tf.nn.conv2d(data, conv1_w, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(conv1 + conv1_b)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv2 = tf.nn.conv2d(pool1, conv2_w, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(conv2 + conv2_b)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv3 = tf.nn.conv2d(pool2, conv3_w, [1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(conv3 + conv3_b)
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        hidden = tf.nn.relu(tf.nn.conv2d(pool3, fc1_w, [1, 1, 1, 1], padding='VALID') + fc1_b)
        hidden_dropout = tf.nn.dropout(hidden, hidden_dprob)
        hidden2 = tf.nn.relu(tf.nn.conv2d(hidden_dropout, fc2_w, [1, 1, 1, 1], padding='VALID') + fc2_b)
        hidden2_dropout = tf.nn.dropout(hidden2, hidden_dprob)
        output = tf.nn.conv2d(hidden2_dropout, fc3_w, [1, 1, 1, 1], padding='VALID') + fc3_b
        return tf.reshape(output, [-1, num_labels])

    def test_model(data):
        # 3 basic convolution and pooling layers followed by 3 fully convolutional ILO fully connected layers
        # Note: Dropout must be removed for inference
        conv1 = tf.nn.conv2d(data, conv1_w, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(conv1 + conv1_b)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv2 = tf.nn.conv2d(pool1, conv2_w, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(conv2 + conv2_b)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv3 = tf.nn.conv2d(pool2, conv3_w, [1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(conv3 + conv3_b)
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        hidden = tf.nn.relu(tf.nn.conv2d(pool3, fc1_w, [1, 1, 1, 1], padding='VALID') + fc1_b)
        hidden2 = tf.nn.relu(tf.nn.conv2d(hidden, fc2_w, [1, 1, 1, 1], padding='VALID') + fc2_b)
        output = tf.nn.conv2d(hidden2, fc3_w, [1, 1, 1, 1], padding='VALID') + fc3_b
        if upsample:
            print output.get_shape().as_list()
            scaled_up_preds = tf.image.resize_images(output, img_height, img_width)
            logits = tf.reshape(scaled_up_preds, [-1, num_labels])
            logits = tf.nn.softmax(logits)
            return logits
        else:
            return tf.reshape(output, [-1, num_labels])

    if train:
        logits = train_model(images)
    else:
        logits = test_model(images)

    # Saver used to save only the convolutional layers for transfer learning
    ss_saver = tf.train.Saver({"conv1_w": conv1_w, "conv1_b": conv1_b, "conv2_w": conv2_w, "conv2_b": conv2_b,
                               "conv3_w": conv3_w, "conv3_b": conv3_b})

    return logits, ss_saver


def run_training(lr, num_epochs, path, restore_path):
    """
    Run the training for the model
    :param lr: The learning rate
    :param num_epochs: number of full-passes through the training data
    :param path: Path to save/restore the model
    :param restore_path: Path to restore the convolutional layers for semantic segmentation
    """
    with tf.Graph().as_default():

        # Pull in the images using the defined readers and perform distortions for training
        train_images, train_labels = distorted_inputs(train_data, num_threads=10)

        # Get predictions and variables to save for semantic segmentation
        logits, ss_saver = inference(train=True, images=train_images)

        # Calculate the loss between predictions and labels
        loss = calc_loss(logits, train_labels)

        # Perform gradient descent
        train_op = training(loss, learning_rate=lr)

        # Initialize saver and session
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        # Restore or initialize the model parameters
        if os.path.isfile(path):
            saver.restore(sess=sess, save_path=path)
            print 'Model Restored'
        else:
            # Restore the previously saved variables after initizalization to over-write these variables
            sess.run(tf.initialize_all_variables())
            print 'Model Initialized'
            ss_saver.restore(sess=sess, save_path=restore_path)
            print 'Semantic Segmentation Variables Restored'

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

        # After done training, calculate accuracy on test data.
        print "===================================="
        print "Test accuracy: ", evaluate(test_data, path)


def semantic_segmentation(lr, num_epochs):
    """
    Run the semantic_segmentation model
    :param lr: The learning rate
    :param num_epochs: The number of epochs
    """
    checkpoint_file = './sem_seg/sem_seg.ckpt'
    wd = os.getcwd()
    if not os.path.exists(wd + '/sem_seg'):
        os.mkdir(wd + '/sem_seg')

    run_training(lr, num_epochs, checkpoint_file, './base/sem_seg.ckpt')