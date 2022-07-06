from unittest import result
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import Metric
import numpy as np

# tf.compat.v1.disable_eager_execution()


class Client:
    def __init__(self, clipping_threshold, noise_multiplier, lr, train_data, train_label, test_data, test_label, seed, local_updates) -> None:
        self.seed = seed
        tf.random.set_seed(self.seed)
        self.clipping_threshold = clipping_threshold
        self.noise_multiplier = noise_multiplier
        self.lr = lr

        self.model = tf.keras.Sequential([
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

        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.cross_ent = tf.compat.v1.losses.softmax_cross_entropy
        self.model.compile(optimizer=self.optimizer, loss=self.cross_ent, metrics=['accuracy'])
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.std_dev = 1
        self.local_updates = local_updates

        # reset random seed for gaussian noise
        seed = np.random.randint(0, 1000)
        tf.random.set_seed(seed)


    def batch_clip(self, grads):
        batch_size = grads[0].shape[0]

        global_norms = tf.norm([ tf.norm(tf.reshape(w, (batch_size, -1)), axis=1) for w in grads], axis=0)
        ratios = tf.math.reciprocal(global_norms) * tf.clip_by_value(global_norms, 0.0, self.clipping_threshold)

        clipped_grad = [ tf.einsum("i...,i->i...", w, ratios) for w in grads ]

        return clipped_grad

    def batch_noising(self, grads):
        for i in range(len(grads)):
            noise = tf.random.normal(shape=tf.shape(grads[i]), mean=0.0, stddev=self.std_dev * self.noise_multiplier * self.clipping_threshold, dtype=tf.float32)
            grads[i] += noise
        return grads

    def batch_mean(self, grads):
        for i in range(len(grads)):
            grads[i] = tf.reduce_mean(grads[i], 0)

        return grads

    def client_update(self, batch_size):
        acc_grads = []
        for _ in range(self.local_updates):
            # self.model.evaluate(self.test_data, self.test_label)
            indices = np.random.randint(0, len(self.train_data), size=batch_size)
            selected_labels = [self.train_label[indices[i]].reshape(10) for i in range(len(indices))]

            with tf.GradientTape() as tape:
                selected_input = [self.train_data[indices[i]].reshape(1, 28, 28, 1) for i in range(len(indices))]
                stacked_input  = tf.reshape(selected_input, [batch_size, 28, 28, 1])
                selected_y = self.model(stacked_input)
                loss = self.cross_ent(selected_labels, selected_y, reduction=tf.compat.v1.losses.Reduction.NONE)
            grads = tape.jacobian(loss, self.model.trainable_variables)

            del tape

            grads = self.batch_clip(grads)
            grads = self.batch_noising(grads)
            mean_grad = self.batch_mean(grads)
            self.optimizer.apply_gradients(zip(mean_grad, self.model.trainable_variables))
            acc_grads.append(mean_grad)

        return np.sum(acc_grads, axis=0)

    def apply_gradient(self, gradient):
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))


    def assign_parameters(self, parameters):
        for i in range(len(parameters)):
            self.model.trainable_variables[i].assign(parameters[i])