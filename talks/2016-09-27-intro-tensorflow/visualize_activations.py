from models.base_convnet import inference
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os

IMAGE_SIZE = 24


def plotNNFilter(units):
    """
    Function to plot a certain layer
    :param units: convnet layer
    """
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    for i in xrange(0, filters):
        plt.subplot(7, 6, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")
    plt.tight_layout(pad=3.0)
    plt.show()


def run_model_image(checkpoint_file, image):
    """
    Run an image through the trained model and vizualize its activations
    :param checkpoint_file: The saved model parameters for the basic model
    :param image: The supplied image (same dimensions as training).
    """
    with tf.Graph().as_default():
        image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])
        image = tf.image.per_image_whitening(image)
        image = tf.reshape(image, [1, IMAGE_SIZE, IMAGE_SIZE, 1])
        image = tf.cast(image, tf.float32)

        relu1, relu2, relu3 = inference(train=False, images=image, visualize=True)

        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()
        saver.restore(sess=sess, save_path=checkpoint_file)

        units = relu1.eval(session=sess)
        plotNNFilter(units)

        units = relu2.eval(session=sess)
        plotNNFilter(units)

        units = relu3.eval(session=sess)
        plotNNFilter(units)


def plot_activations(image_dir):
    """
    Plot the activations for a given image
    :param checkpoint_file: Where the model is saved
    :param image_dir:
    """
    read_image = cv2.imread(image_dir, 0)
    read_image = cv2.resize(read_image, (24, 24), interpolation=cv2.INTER_AREA)

    run_model_image(checkpoint_file='./base/base.ckpt', image=read_image)


def visualize_activations():
    # Get the test image directory and then run the test images through the model
    # and visualize activations.
    wd = os.getcwd()
    test_images = wd + '/test_data/number_samples/'

    for image_dir in os.listdir(test_images):
        print image_dir
        plot_activations(test_images + image_dir)