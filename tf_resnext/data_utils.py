from functools import reduce

import numpy as np
import tensorflow as tf


def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = [x.astype(np.float32) for x in [x_train, x_test]]
    y_train, y_test = [y.flatten() for y in [y_train, y_test]]
    mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std = x_train.std(axis=(0, 1, 2), keepdims=True)
    x_train, x_test = [(x - mean) / std for x in [x_train, x_test]]
    return [(tf.data.Dataset.from_tensor_slices((x, y)), len(x))
            for x, y in [(x_train, y_train), (x_test, y_test)]]


def random_pad_and_crop(image, label, pad_size=4):
    return (
        tf.image.random_crop(
            tf.pad(image, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], constant_values=0),
            tf.shape(image),
        ),
        label,
    )


def random_flip_left_right(image, label):
    return tf.image.random_flip_left_right(image), label


def preprocess_dataset(
        dataset,
        size,
        batch_size,
        map_fns=(),
        map_fn_num_threads=10,
        shuffle_buffer_size=100000,
        prefetch_buffer_size=100,
):
    dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda image, label: reduce(lambda value, fn: fn(*value), map_fns, (image, label)),
        num_parallel_calls=map_fn_num_threads,
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    size //= batch_size
    dataset = dataset.repeat()
    dataset = dataset.prefetch(prefetch_buffer_size)
    return dataset, size
