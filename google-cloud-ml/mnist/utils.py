#!/usr/bin/python3

import os
import json
import argparse
import numpy as np
import tensorflow as tf

def load_mnist_data(local=False):
    """Prepare data for classification"""

    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # reshape data
    train_images = train_images.reshape(len(train_images), -1)
    test_images = test_images.reshape(len(test_images), -1)
    train_labels = np.asarray(train_labels).astype('int').reshape((-1, 1))
    test_labels = np.asarray(test_labels).astype('int').reshape((-1, 1))

    # scale values to a range of 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if local:
        # avoid error 'Allocation exceeds 10% of system memory'
        train_images = train_images[:10000]
        train_labels = train_labels[:10000]
        test_images = test_images[:1000]
        test_labels = test_labels[:1000]

    return (train_images, train_labels), (test_images, test_labels)

def feed_data(features, labels, batch_size, shuffle=True):
    """Feet data to specs"""

    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    # convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)

    dataset_iterator = dataset.make_one_shot_iterator()

    return dataset_iterator.get_next()

def serving_input():
    """Defines the features to be passed to the model during inference"""

    features = tf.placeholder(tf.float32, [None, 784])
    return tf.estimator.export.TensorServingInputReceiver(features, features)

def store_test_samples(output_file, n=10):
    """Extract n test samples and store them in a json file"""

    (_, _), (test_images, test_labels) = load_mnist_data()

    ids = np.random.randint(0, len(test_images) + 1, n)
    image_selection = [list(test_images[i]) for i in ids]
    label_selection = [test_labels[i] for i in ids]

    with open(output_file, 'w') as ostream:
        for image in image_selection:
            ostream.write(json.dumps(image) + '\n')

    return label_selection

def get_arguments():
    """Argument parser"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='GCS location for current job')
    parser.add_argument(
        '--shallow',
        help='Whether to use a shallow or a deep neural network',
        action='store_true')
    parser.add_argument(
        '--train-steps',
        type=int,
        help="Number of steps to run training job for",
        default=10)
    parser.add_argument(
        '--eval-steps',
        type=int,
        help='Number of steps to run evalution job for (at each checkpoint)',
        default=10)
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=128)
    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=128)
    parser.add_argument(
        '--verbosity',
        default='INFO',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'])
    parser.add_argument(
        '--local',
        help='Whether the current execution is run in local mode',
        action='store_true')

    return parser.parse_args()
