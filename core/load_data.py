'''
@description: 
@author: liutaiting
@lastEditors: liutaiting
@Date: 2020-04-03 10:01:59
@LastEditTime: 2020-04-03 10:37:58
'''

import pandas as pd
import numpy as np
from config import NUM_CLASSES, image_width, image_height, channels, BATCH_SIZE
import tensorflow as tf

def fer2013_load_data():
    '''
    @description: 读取fer2013数据集
    @param {type} 
    @return: train_flow, val_flow, num_train, num_val
    '''
    
    data = pd.read_csv('./data/fer2013.csv')

    # print(data.head())
    # print(data.emotion.value_counts())

    # 选取train set和val set
    train_set = data[(data.Usage == 'Training')] 
    val_set = data[(data.Usage == 'PublicTest')]
    test_set = data[(data.Usage == 'PrivateTest')]
    
    # 划分dataset
    X_train = np.array(list(map(str.split, train_set.pixels)), np.float32) 
    X_val = np.array(list(map(str.split, val_set.pixels)), np.float32) 
    X_test = np.array(list(map(str.split, test_set.pixels)), np.float32) 

    # 将dataset进行reshape ——> (nums,width,heigh,channel)
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1) 
    X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

    num_train = X_train.shape[0]
    num_val = X_val.shape[0]
    num_test = X_test.shape[0]

    # 获取labels
    y_train = tf.keras.utils.to_categorical(train_set.emotion, NUM_CLASSES) 
    y_val = tf.keras.utils.to_categorical(val_set.emotion, NUM_CLASSES) 
    y_test = tf.keras.utils.to_categorical(test_set.emotion , NUM_CLASSES)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator( 
    rescale=1./255,
    rotation_range = 10,
    horizontal_flip = True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode = 'nearest')

    testgen = tf.keras.preprocessing.image.ImageDataGenerator( 
    rescale=1./255
    )

    datagen.fit(X_train)
    testgen.fit(X_test)

    train_flow = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE) 
    val_flow = testgen.flow(X_val, y_val, batch_size=BATCH_SIZE) 
    test_flow = testgen.flow(X_test, y_test, batch_size=BATCH_SIZE) 
    
    return train_flow, val_flow, num_train, num_val