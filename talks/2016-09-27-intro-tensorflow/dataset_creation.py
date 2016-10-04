import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define the number of test and validation records
num_test = 10000
num_valid = 5000


def create_tensorflow_files(directory, dataset, labels):
    """
    Randomize the data and labels to make sure there is a random splitting for train/test/validation sets. Then
    feed to the tensorflow_conversion function to be outputted in the supplied directory.
    :param directory: The path for the data directory
    :param dataset: The image_data to be outputted
    :param labels: The labels associated with the supplied dataset
    """
    # Find a random permutation and shuffle the dataset accordingly
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]

    # Split the image and label numpy arrays into test/train/valid datasets
    test_dataset = shuffled_dataset[:num_test, :, :]
    valid_dataset = shuffled_dataset[num_test:num_test+num_valid, :, :]
    train_dataset = shuffled_dataset[num_test+num_valid:, :, :]

    test_labels = shuffled_labels[:num_test]
    valid_labels = shuffled_labels[num_test:num_test+num_valid]
    train_labels = shuffled_labels[num_test+num_valid:]

    # Output the bincounts to make sure there is a relatively even distribution of the classes
    print 'Test Bincount:', np.bincount(test_labels)
    print 'Valid Bincount:', np.bincount(valid_labels)
    print 'Train Bincount:', np.bincount(train_labels)

    # Write the output in the tensorflow format
    tensorflow_conversion(train_dataset.astype('uint8'), train_labels, directory, 'train')
    tensorflow_conversion(valid_dataset.astype('uint8'), valid_labels, directory, 'valid')
    tensorflow_conversion(test_dataset.astype('uint8'), test_labels, directory, 'test')


def tensorflow_conversion(images, labels, directory, name):
    """
    Write the images and labels in the TensorFlow binary format.
    :param images: Image numpy array
    :param labels: Label numpy array
    :param directory: The directory path to put the data
    :param name: The name (train/test/valid) for the file
    """
    num_examples = labels.shape[0]
    filename = directory + name + '.tfrecords'
    print 'Writing', filename
    print np.shape(images)
    print np.shape(labels)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[index]])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))
        writer.write(example.SerializeToString())


def write_files(image_dirs, labels, directory, name):
    """
    Write the image path and labels to a .txt file
    :param image_dirs: Image path numpy array
    :param labels: Label numpy array
    :param directory: The data directory
    :param name: The name (test/train/valid)
    """
    with open(directory+name+".txt", "w") as text_file:
        for index in range(0, len(image_dirs)):
            text_file.write(str(labels[index])+','+str(image_dirs[index]))
            text_file.write('\n')


def create_text_files(image_list, label_list, directory):
    """
    Function to create shuffled lists of the path and path labels to display a potential directory
    method to train a tensorflow convnet model
    :param image_list: Numpy list of image paths
    :param label_list: Numpy list of labels
    :param directory: The data output directory
    """
    # Find a random permutation and shuffle the dataset accordingly
    permutation = np.random.permutation(len(image_list))
    shuffled_dataset = image_list[permutation]
    shuffled_labels = label_list[permutation]

    # Split the image and label numpy arrays into test/train/valid datasets
    test_dataset = shuffled_dataset[:num_test]
    valid_dataset = shuffled_dataset[num_test:num_test + num_valid]
    train_dataset = shuffled_dataset[num_test + num_valid:]

    test_labels = shuffled_labels[:num_test]
    valid_labels = shuffled_labels[num_test:num_test + num_valid]
    train_labels = shuffled_labels[num_test + num_valid:]

    # Output the bincounts to make sure there is a relatively even distribution of the classes
    print 'Test Bincount:', np.bincount(test_labels)
    print 'Valid Bincount:', np.bincount(valid_labels)
    print 'Train Bincount:', np.bincount(train_labels)

    # Write the output in the text path format to eventually pull images in real-time
    write_files(image_dirs=test_dataset, labels=test_labels, directory=directory, name='test')
    write_files(image_dirs=valid_dataset, labels=valid_labels, directory=directory, name='valid')
    write_files(image_dirs=train_dataset, labels=train_labels, directory=directory, name='train')


def find_images(directory):
    """
    Find the images in the data directory (assumed to be in directories associated with their label) and create arrays
    that store the path and label. This code can be easily extended to labels that are not numeric. If we wnted to
    run it for example as ['One', 'Two'] we could create an array, do an index match on that index and store that
    as the label.
    :param directory: The data directory with directories named as the labels that also store the images
    :return: A numpy array of the image paths and the image labels
    """
    image_list = []
    label_list = []
    for dir_file in os.listdir(directory):
        if os.path.isdir(directory + dir_file):
            for image in os.listdir(directory + dir_file):
                label_list.append(int(dir_file))
                image_list.append(str(directory + dir_file + '/' + image))

    return np.array(image_list), np.array(label_list)


def create_images(images, image_label, labels, directory):
    """
    This piece of code creates the directory structure to store the images. Since the canonical label is the same
    as the actual label i.e. [0, 1, 2, ..., 9] I save the directory structure as such. It would be very simple to
    create a canonical lookup i.e. ['One', 'Two', ..., 'Nine'] that you can index as it comes back in. I dont do this
    here for the sake of time, but it is easy to do with a lookup array.
    :param images: The image data
    :param labels: The labels
    :param directory: The data directory to output
    """
    for label in labels:
        if not os.path.exists(directory + '/data/' + str(label)):
            os.mkdir(directory + '/data/' + str(label))

    for img_index, img_label in enumerate(image_label):
        image = np.reshape(images[img_index, :, :], (28, 28))

        im = Image.fromarray(image).convert('L')
        im.save(directory + '/data/' + str(img_label) + '/img_' + str(img_index) + '.png')


def create_data():
    # This little piece of code is taken from the mnist tutorial online found here:
    # https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html#load-mnist-data

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    # I dont want to use the raw dataset since I want to create an image database as well as display the native format that
    # TensorFlow uses when outputing I looked at the dataset creation code found in the following link:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

    # They create a DataSet class to hold the mnist data. I am going to extract it and de-regulate it so it will be
    # more representative of a dataset that you might find in the wild and illustrate how one might prep that data for
    # usage by a learning (machine/deep) algorithm through TensorFlow.

    train_image_data = mnist.train._images
    train_image_label = mnist.train._labels

    valid_image_data = mnist.validation._images
    valid_image_label = mnist.validation._labels

    test_image_data = mnist.test._images
    test_image_label = mnist.test._labels

    # The rest of the code is a little counter-intuitive because I am going to create two new formats to illustrate
    # a couple different ways to read the data in.

    image_data = np.vstack([train_image_data, valid_image_data, test_image_data])
    image_label = np.hstack([train_image_label, valid_image_label, test_image_label])

    # Reshape the image to conform with a typical image input as well as what the convents produced later will expect.
    # The data is also pre-normalized which we will do as part of our reader process (thus the multiply by 255).
    image_data = np.multiply(np.reshape(image_data, (-1, 28, 28)), 255)
    image_label = image_label.astype(int)

    # Create the data directory structure. This will have both the .tfrecords files, the .txt files that contain
    # The label associated with the images, and the raw images.
    wd = os.getcwd()
    if not os.path.exists(wd + '/data'):
        os.mkdir(wd + '/data')

    # Create the images in a directory with their label
    create_images(image_data, image_label, np.unique(image_label), wd)

    # Create the .tfrecords data
    create_tensorflow_files(directory=wd+'/data/', dataset=image_data, labels=image_label)

    # Get the data from the file directory and create the .txt files with label,path
    image_list, label_list = find_images(wd + '/data/')
    create_text_files(image_list, label_list, directory=wd+'/data/')
