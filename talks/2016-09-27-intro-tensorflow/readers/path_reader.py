import tensorflow as tf

input_image_size = 28
output_image_size = 24
input_image_channels = 1
num_labels = 10
valid_records = 5000
test_records = 10000
train_records = 55000
batch_size = 100


def read_path_file(image_file):
    f = open(image_file, 'r')
    paths = []
    labels = []
    for line in f:
        label, path = line[:-1].split(',')
        paths.append(path)
        labels.append(int(label))
    return paths, labels


def distorted_inputs(image_file, num_threads):
    image_list, label_list = read_path_file(image_file)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

    filename_queue = tf.train.slice_input_producer([images, labels], shuffle=False)
    result = read_image(filename_queue)

    distorted_image = tf.image.resize_image_with_crop_or_pad(result.image, output_image_size,
                                                             output_image_size)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    white_image = tf.image.per_image_whitening(distorted_image)

    return generate_batches(white_image, result.label, num_threads)


def inputs(image_file, num_threads):
    image_list, label_list = read_path_file(image_file)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

    filename_queue = tf.train.slice_input_producer([images, labels], shuffle=False)
    result = read_image(filename_queue)

    distorted_image = tf.image.resize_image_with_crop_or_pad(result.image, output_image_size,
                                                             output_image_size)
    white_image = tf.image.per_image_whitening(distorted_image)

    return generate_batches(white_image, result.label, num_threads)


def read_image(filename_queue):
    class image_object(object):
        pass
    result = image_object()

    file_contents = tf.read_file(filename_queue[0])
    image = tf.image.decode_png(file_contents, channels=input_image_channels)
    image = tf.cast(image, tf.float32)
    result.image = tf.reshape(image, [input_image_size, input_image_size, input_image_channels])

    label = tf.cast(filename_queue[1], tf.int32)
    result.label = tf.sparse_to_dense(label, [num_labels], 1.0, 0.0)

    return result


def generate_batches(image, label, num_threads):

    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size,
        num_threads=num_threads,
    )

    return images, labels