import os, sys, tarfile
import numpy as np

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib
else:
    import urllib

try:
    from imageio import imsave
except:
    from scipy.misc import imsave


HEIGHT = 96
WIDTH = 96
DEPTH = 3
SIZE = HEIGHT * WIDTH * DEPTH

DATA_DIR = '../STL'
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
DATA_PATH = '../STL/stl10_binary/'


def read_train_data():
    data = read_all_images(DATA_PATH + 'train_X.bin')
    labels = read_labels(DATA_PATH + 'train_y.bin')

    return data, labels


def read_vali_data():
    data = read_all_images(DATA_PATH + 'test_X.bin')
    labels = read_labels(DATA_PATH + 'test_y.bin')

    return data, labels


def read_labels(path_to_labels):

    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)

        return labels-1.0


def read_all_images(path_to_data):

    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)

        images = np.reshape(everything, (-1, 3, 96, 96))

        images = np.transpose(images, (0, 3, 2, 1))

        images = images/255.0

        return images


def random_crop_and_flip(batch_data, padding_size):

    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    batch_data = np.pad(batch_data, pad_width=pad_width, mode='constant', constant_values=0)

    cropped_batch = np.zeros(len(batch_data) * HEIGHT * WIDTH * DEPTH).reshape(len(batch_data), HEIGHT, WIDTH, DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+HEIGHT, y_offset:y_offset+WIDTH, :]
        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch


def horizontal_flip(image, axis):

    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = np.flip(image, axis)

    return image


def whitening_image(image_np):

    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        std = np.max([np.std(image_np[i, ...]), 1.0 / np.sqrt(HEIGHT * WIDTH * DEPTH)])
        image_np[i, ...] = (image_np[i, ...] - mean) / std

    return image_np