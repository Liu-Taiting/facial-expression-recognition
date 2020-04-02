'''
@description: 
@author: liutaiting
@lastEditors: liutaiting
@Date: 2020-04-02 23:18:17
@LastEditTime: 2020-04-02 23:19:53
'''

import tensorflow as tf
from config import *


def VGG19():
    model = tf.keras.Sequential()

    # block1
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu',
                                     input_shape=(image_height,image_width,channels)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                                           strides=2,
                                           padding='same'))

    # block2
    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                                           strides=2,
                                           padding='same'))

    # block3
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                                           strides=2,
                                           padding='same'))

    # block4
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                                           strides=2,
                                           padding='same'))

    # block5
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512,
                                     kernel_size=(3,3),
                                     strides=1,
                                     padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),
                                           strides=2,
                                           padding='same'))

    # block6->fc layers
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(units=512,
    #                                 activation='relu'))
    # model.add(tf.keras.layers.Dense(units=4096,
    #                                 activation='relu'))
    model.add(tf.keras.layers.Dense(units=NUM_CLASSES,
                                    activation='softmax'))

    return model