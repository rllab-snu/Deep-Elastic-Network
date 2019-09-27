import cv2
import numpy as np

IMAGE_W = 64
IMAGE_H = 64
IMAGE_D = 3
dataset_root = '../imagenet-tiny'


def read_train_data():
    data_size = 100000
    imgset = np.array(np.zeros(data_size * IMAGE_H * IMAGE_W * IMAGE_D, dtype=np.float32)).reshape([data_size, IMAGE_H, IMAGE_W, IMAGE_D])
    labset = np.array(np.zeros(data_size))
    with open(dataset_root + '/train.txt', 'r') as f:
        for i in range(data_size):
            line = f.readline()
            tmp = line.split()
            img = cv2.imread(dataset_root + tmp[0])
            img = img.astype(np.float32)
            img = img/255

            img = img.reshape([int(IMAGE_H), int(IMAGE_W), IMAGE_D])
            img = cv2.resize(img, dsize=(IMAGE_H, IMAGE_W))
            img = img.reshape([1, IMAGE_H, IMAGE_W, IMAGE_D])

            imgset[i] = img
            labset[i] = tmp[1]

    return imgset, labset


def read_vali_data():
    data_size = 10000
    imgset = np.array(np.zeros(data_size * IMAGE_H * IMAGE_W * IMAGE_D, dtype=np.float32)).reshape([data_size, IMAGE_H, IMAGE_W, IMAGE_D])
    labset = np.array(np.zeros(data_size))
    with open(dataset_root + '/test.txt', 'r') as f:
        for i in range(data_size):
            line = f.readline()
            tmp = line.split()
            img = cv2.imread(dataset_root + tmp[0])
            img = img.astype(np.float32)
            img = img/255

            img = img.reshape([int(IMAGE_H), int(IMAGE_W), IMAGE_D])
            img = cv2.resize(img, dsize=(IMAGE_H, IMAGE_W))
            img = img.reshape([1, IMAGE_H, IMAGE_W, IMAGE_D])

            imgset[i] = img
            labset[i] = tmp[1]

    return imgset, labset


def horizontal_flip(image, axis):
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = np.flip(image, axis)

    return image


def whitening_image(image_np):
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMAGE_H * IMAGE_W * IMAGE_D)])
        image_np[i, ...] = (image_np[i, ...] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size):

    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    batch_data = np.pad(batch_data, pad_width=pad_width, mode='constant', constant_values=0)

    cropped_batch = np.zeros(len(batch_data) * IMAGE_H * IMAGE_W * IMAGE_D).reshape(len(batch_data), IMAGE_H, IMAGE_W, IMAGE_D)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMAGE_H, y_offset:y_offset+IMAGE_W, :]
        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch
