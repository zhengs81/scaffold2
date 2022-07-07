from unittest import result
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import Metric
import copy
import os
from numpy import linalg as LA

import numpy as np
import dill

from client import Client
from sklearn.utils import shuffle

# tf.compat.v1.disable_eager_execution()


def partition_data(train_data, train_labels):
    p_train_data = [[], [], [], [], [], [], [], [], [], []]
    p_train_label = [[], [], [], [], [], [], [], [], [], []]

    for i in range(len(train_labels)):
        label_idx = np.nonzero(train_labels[i])[0][0]
        p_train_data[label_idx].append(train_data[i])
        p_train_label[label_idx].append(train_labels[i])

    return p_train_data, p_train_label


def partially_shuffle_data(train_data, train_labels, percentage):
    num_data = len(train_data)
    num_fixed = int(num_data * percentage)
    fixed_train_data = train_data[0: num_fixed]
    fixed_train_label = train_labels[0: num_fixed]

    p_train_data, p_train_label = partition_data(fixed_train_data, fixed_train_label)

    rest_train_data = train_data[num_fixed: ]
    rest_train_label = train_labels[num_fixed: ]

    rest_train_data, rest_train_label = shuffle(rest_train_data, rest_train_label)

    rest_len = int(len(rest_train_data) / 10)

    for i in range(10):
        p_train_data[i] = np.concatenate((p_train_data[i], rest_train_data[i*rest_len: (i+1)*rest_len]))
        p_train_label[i] = np.concatenate((p_train_label[i], rest_train_label[i*rest_len: (i+1)*rest_len]))

    return p_train_data, p_train_label


if __name__ == '__main__':
    SEED = 0
    NUM_CLIENTS = 10
    CLIENT_LOCAL_UPDATES = 5
    turn = 0
    SHUFFLE_RATE = 0.2
    tf.random.set_seed(SEED)

    train, test = tf.keras.datasets.mnist.load_data()

    train_data, train_labels = train
    test_data, test_labels = test

    # train_data = train_data[0: 20000]
    # train_labels = train_labels[0: 20000]

    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

    train_labels = np.array(train_labels, dtype=np.int32) # [0, 4, 7, 8]
    
    test_labels = np.array(test_labels, dtype=np.int32)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10) # convert into one-hot format
    
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    # partition the train data into 10 sets
    train_data, train_labels = partially_shuffle_data(train_data, train_labels, SHUFFLE_RATE)
    # print(train_labels[9])

    epochs = 1
    l2_norm_clip = 1.3
    std_dev = 1.0
    learning_rate = 0.004
    noise_multiplier = 0.3 # 0.01
    num_microbatch = 200
    overall_batch = 60000

    clients = []

    acc_arr = []

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 8,
                            strides=2,
                            padding='same',
                            activation='relu',
                            input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Conv2D(32, 4,
                            strides=2,
                            padding='valid',
                            activation='relu'),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    cross_ent = tf.compat.v1.losses.softmax_cross_entropy
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=cross_ent, metrics=['accuracy'])

    for i in range(NUM_CLIENTS):
        clients.append(Client(l2_norm_clip, noise_multiplier, learning_rate, train_data[i], train_labels[i], test_data, test_labels, SEED, CLIENT_LOCAL_UPDATES))

    for _ in range(epochs):
        for _ in range(int(overall_batch / (num_microbatch * CLIENT_LOCAL_UPDATES))):
            # model.compile(optimizer=optimizer, loss=cross_ent, metrics=['accuracy'])
            results = model.evaluate(test_data, test_labels)
            acc_arr.append(results[1])

            turn += 1

            print(f"test accuracy {results[1]} for turn {turn}")

            grads = []

            for i in range(len(clients)):
                clients[i].assign_parameters(model.trainable_variables)
                grads.append(clients[i].client_update(num_microbatch))

            grads = np.mean(grads, axis=0)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))


    filename = "Non-iid,Partition,clip&noise, l2_norm=" + str(l2_norm_clip) + "noise_multiplier=" + str(noise_multiplier) + "epochs=" + str(epochs) + "num_microbatch" + str(num_microbatch) + "local_updates=" + str(CLIENT_LOCAL_UPDATES) + "shuffle_rate=" + str(SHUFFLE_RATE)

    with open(filename, "wb") as dill_file:
        dill.dump(acc_arr, dill_file)

