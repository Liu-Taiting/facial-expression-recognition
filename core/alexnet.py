'''
@description: 
@author: liutaiting
@lastEditors: liutaiting
@Date: 2020-04-01 23:15:44
@LastEditTime: 2020-04-02 23:47:12
'''

import tensorflow as tf
from config import NUM_CLASSES, image_width, image_height, channels


def Alexnet():
    model = tf.keras.Sequential([
        # layer 1
        tf.keras.layers.Conv2D(filters=96,
                               kernel_size=(11, 11),
                               strides=4,
                               padding='valid',
                               activation='relu',
                               input_shape=(image_width, image_height, channels)),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding='valid'),
        tf.keras.layers.BatchNormalization(),
        # layer 2
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=1,
                               padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding='same'),
        tf.keras.layers.BatchNormalization(),
        # layer 3
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(5, 5),
                               strides=1,
                               padding='same',
                               activation='relu'),
        # layer 4
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3,3),
                               strides=1,
                               padding='same',
                               activation='relu'),
        # layer 5
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               strides=1,
                               padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding='same'),
        tf.keras.layers.BatchNormalization(),
        # layer 6
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512,
                              activation="relu"),
        tf.keras.layers.Dropout(rate=0.5),
        # # layer 7
        # tf.keras.layers.Dense(units=4096,
        #                       activation="relu"),
        # tf.keras.layers.Dropout(rate=0.2),
        # layer 8
        tf.keras.layers.Dense(units=NUM_CLASSES,
                              activation=tf.keras.activations.softmax)
    ])

    return model
# model = Alexnet()
# model.summary()