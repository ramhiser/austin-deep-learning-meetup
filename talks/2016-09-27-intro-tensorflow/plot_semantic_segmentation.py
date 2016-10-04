import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from models.semantic_segmentation import inference

# Image dimensions for the file we are providing the model
base_width = 336
base_height = 140

# Model variables
image_size = 24
num_labels = 10


def run_model_image(image, trans_height, trans_width):
    """
    Function to load the smeantic segmentation graph and run the image through.
    :param image: Image provided
    :return: Up-sampled predictions
    """
    with tf.Graph().as_default():
        image = tf.reshape(image, [trans_height, trans_width, 1])
        image = tf.image.per_image_whitening(image)
        image = tf.reshape(image, [1, trans_height, trans_width, 1])
        image = tf.cast(image, tf.float32)

        train_prediction, _ = inference(train=False, images=image, upsample=True, img_height=base_height,
                                        img_width=base_width)

        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()
        saver.restore(sess=sess, save_path='./sem_seg/sem_seg.ckpt')
        predictions = sess.run(train_prediction)

        return predictions


def create_ss_plots():
    # Tell the model which pixel heights to run between and how many pixels to skip
    ratio = (336/140.0)
    start = 80
    increment = 2
    end = 120 + increment

    final_hm = []
    # Iterate through the various image sizes
    for hght in range(start, end, increment):

        # Run the image through in the provided size
        trans_height = hght
        trans_width = int(trans_height * ratio)

        read_image = cv2.imread('./test_data/all_numbers.png', 0)
        read_image = cv2.resize(read_image, (trans_width, trans_height), interpolation=cv2.INTER_AREA)
        print np.shape(read_image)

        in_imdata = read_image.reshape((1, trans_height, trans_width, 1))
        heatmap = run_model_image(image=in_imdata, trans_height=trans_height, trans_width=trans_width)

        # Reshape for the image and only keep labels above a low threshold to avoid too much noise
        heatmap = heatmap.reshape(base_height, base_width, num_labels)
        heatmap[heatmap <= 0.2] = 0.2
        print hght

        if hght == start:
            final_hm = heatmap
        else:
            final_hm += heatmap

    # Create the output plot
    fig, ax = plt.subplots(3, 4)
    ax[0, 0].imshow(read_image)
    ax[0, 0].set_title('Image')
    ax[0, 1].imshow(final_hm[:, :, 0], vmin=np.min(final_hm[:, :, 0]), vmax=np.max(final_hm[:, :, 0]))
    ax[0, 1].set_title('Zero')
    ax[0, 2].imshow(final_hm[:, :, 1], vmin=np.min(final_hm[:, :, 1]), vmax=np.max(final_hm[:, :, 1]))
    ax[0, 2].set_title('One')
    ax[0, 3].imshow(final_hm[:, :, 2], vmin=np.min(final_hm[:, :, 2]), vmax=np.max(final_hm[:, :, 2]))
    ax[0, 3].set_title('Two')
    ax[1, 0].imshow(final_hm[:, :, 3], vmin=np.min(final_hm[:, :, 3]), vmax=np.max(final_hm[:, :, 3]))
    ax[1, 0].set_title('Three')
    ax[1, 1].imshow(final_hm[:, :, 4], vmin=np.min(final_hm[:, :, 4]), vmax=np.max(final_hm[:, :, 4]))
    ax[1, 1].set_title('Four')
    ax[1, 2].imshow(final_hm[:, :, 5], vmin=np.min(final_hm[:, :, 5]), vmax=np.max(final_hm[:, :, 5]))
    ax[1, 2].set_title('Five')
    ax[1, 3].imshow(final_hm[:, :, 6], vmin=np.min(final_hm[:, :, 6]), vmax=np.max(final_hm[:, :, 6]))
    ax[1, 3].set_title('Six')
    ax[2, 0].imshow(final_hm[:, :, 7], vmin=np.min(final_hm[:, :, 7]), vmax=np.max(final_hm[:, :, 7]))
    ax[2, 0].set_title('Seven')
    ax[2, 1].imshow(final_hm[:, :, 8], vmin=np.min(final_hm[:, :, 8]), vmax=np.max(final_hm[:, :, 8]))
    ax[2, 1].set_title('Eight')
    ax[2, 2].imshow(final_hm[:, :, 9], vmin=np.min(final_hm[:, :, 9]), vmax=np.max(final_hm[:, :, 9]))
    ax[2, 2].set_title('Nine')
    plt.show()
