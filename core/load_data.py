'''
@description: fer2013导入ok，ck+随后改成onehot形式
@author: liutaiting
@lastEditors: liutaiting
@Date: 2020-04-03 10:01:59
@LastEditTime: 2020-04-04 23:43:21
'''

import pandas as pd
import numpy as np
import pathlib
import random
from config import NUM_CLASSES, image_width, image_height, channels, BATCH_SIZE
import tensorflow as tf

def fer2013_load_data():
    '''
    @description: 读取fer2013数据集
    @param {type} 
    @return: train_flow, val_flow, X_test, y_test, num_train, num_val
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
    
    return train_flow, val_flow, X_test, y_test, num_train, num_val

def ck_load_data():
    # 过后修改，改成onehot形式

    data_dir = './data/CK+48'
    data_root = pathlib.Path(data_dir)

    # for item in data_root.iterdir():
    #     print(item)

    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    classname = np.array(label_names)
    label_to_index = dict((name, index) for index,name in enumerate(label_names))
    image_count = len(all_image_paths)

    random.shuffle(all_image_paths)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    test_count = int(image_count*0.3)
    train_count = image_count - test_count
    train_data = image_label_ds.skip(test_count)
    test_data = image_label_ds.take(test_count)
    test_label = label_ds.take(test_count)

    y_true = []
    for a,b in test_data:
        y_true.append(b.numpy())
    y_true = np.array(y_true)

    train_data = train_data.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=train_count))
    train_data = train_data.batch(BATCH_SIZE)
    train_data = train_data.prefetch(buffer_size=AUTOTUNE)

    test_data = test_data.batch(BATCH_SIZE)

    return train_data, test_data,train_count,test_count,classname,y_true


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [48, 48])
    image = tf.cast(image, tf.float32)
    image = image/255.0  # normalize to [0,1] range
    return image