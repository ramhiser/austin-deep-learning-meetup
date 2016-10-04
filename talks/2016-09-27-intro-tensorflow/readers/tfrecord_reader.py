import tensorflow as tf

input_image_size = 28
output_image_size = 24
input_image_channels = 1
num_labels = 10
valid_records = 5000
test_records = 10000
train_records = 55000
batch_size = 100


def distorted_inputs(filename, num_threads):

    filename_queue = tf.train.string_input_producer([filename])
    result = read_data(filename_queue)

    #distorted_image = tf.image.random_crop(result.image, [output_image_size, output_image_size])
    distorted_image = tf.random_crop(result.image, [output_image_size, output_image_size,
                                                    input_image_channels])

    # distorted_image = tf.image.random_flip_left_right(distorted_image)
    # distorted_image = tf.image.random_flip_up_down(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    white_image = tf.image.per_image_whitening(distorted_image)

    return generate_batches(white_image, result.label, num_threads)


def inputs(filename, num_threads):

    filename_queue = tf.train.string_input_producer([filename])
    result = read_data(filename_queue)

    distorted_image = tf.image.resize_image_with_crop_or_pad(result.image, output_image_size,
                                                             output_image_size)
    white_image = tf.image.per_image_whitening(distorted_image)

    return generate_batches(white_image, result.label, num_threads)


def read_data(filename_queue):

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        #dense_keys=['image_raw', 'label'],
        #dense_types=[tf.string, tf.int64]
        features={'image_raw': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)}
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([input_image_size * input_image_size * input_image_channels])
    image = tf.cast(image, tf.float32)
    result.image = tf.reshape(image, [input_image_size, input_image_size, input_image_channels])

    label = tf.cast(features['label'], tf.int32)
    result.label = tf.sparse_to_dense(label, [num_labels], 1.0, 0.0)

    return result


def generate_batches(image, label, num_threads):

    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size,
        num_threads=num_threads,
    )

    return images, labels
