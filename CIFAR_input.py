import numpy as np

full_data_dir = '../cifar100/cifar-100-python/train'
vali_dir = '../cifar100/cifar-100-python/test'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 100

NUM_TRAIN_BATCH = 1
EPOCH_SIZE = 50000 * NUM_TRAIN_BATCH


def _read_one_batch(path):
    dicts = np.load(path, encoding='latin1')

    data = dicts['data']
    label = np.array(dicts['fine_labels'])

    return data, label


def read_in_all_images(address_list, shuffle=True):
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([])

    for address in address_list:
        batch_data, batch_label = _read_one_batch(address)
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = len(label)

    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    if shuffle is True:
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)

    return data, label


def horizontal_flip(image, axis):

    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = np.flip(image, axis)

    return image


def whitening_image(image_np):

    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        std = np.max([np.std(image_np[i, ...]), 1.0 / np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i, ...] = (image_np[i, ...] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size):

    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    batch_data = np.pad(batch_data, pad_width=pad_width, mode='constant', constant_values=0)

    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset + IMG_HEIGHT,
                                y_offset:y_offset + IMG_WIDTH, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch


def read_train_data():
    path_list = []
    path_list.append(full_data_dir)
    data, label = read_in_all_images(path_list)

    return data, label


def read_vali_data():
    validation_array, validation_labels = read_in_all_images([vali_dir])

    return validation_array, validation_labels
