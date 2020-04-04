'''
@description: 
@author: liutaiting
@lastEditors: liutaiting
@Date: 2020-04-02 23:23:50
@LastEditTime: 2020-04-04 23:41:07
'''
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from config import NUM_CLASSES, image_width, image_height, channels, BATCH_SIZE, EPOCHS
import core


def main(dataset='fer2013', plot=True, cm=True):
    model = core.Alexnet()
    model.summary()

    if dataset == 'fer2013':
        train_flow, val_flow, X_test, y_test, num_train, num_val = core.fer2013_load_data()
        classes = np.array(("Angry", "Disgust", "Fear",
                            "Happy", "Sad", "Surprise", "Neutral"))

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
                      metrics=['accuracy'])

        history = model.fit_generator(train_flow,
                                      steps_per_epoch=num_train / BATCH_SIZE,
                                      epochs=EPOCHS,
                                      verbose=2,
                                      validation_data=val_flow,
                                      validation_steps=num_val / BATCH_SIZE)
    elif dataset == 'ck+':
        train_data, test_data, train_count, test_count,classes,y_true = core.ck_load_data()
        steps_per_epoch = train_count//BATCH_SIZE
        validation_steps = test_count//BATCH_SIZE

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.9, nesterov=True),
                      metrics=['accuracy'])

        history = model.fit(train_data,
                            epochs=EPOCHS,
                            verbose=2,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=test_data,
                            validation_steps=validation_steps)
        # 后续改
        X_test = test_data
        y_test = y_true
    else:
        return print('wrong dataset')

    if plot:
        core.plot_loss(history, 'Fer2013 train_loss vs val_loss')
        core.plot_acc(history, title='Fer2013 train_acc vs val_acc')
    if cm:
        confusion_matrix(model, X_test, y_test, classes,dataset)
    plt.show()


def confusion_matrix(model, data, lables, classe,dataset):
    if dataset == 'fer2013':
        y_pred_ = model.predict(data/255., verbose=0)
        y_pred = np.argmax(y_pred_, axis=1)
        lables = np.argmax(lables, axis=1)
        core.plot_confusion_matrix(lables, y_pred, classes=classe, normalize=True)
    elif dataset == 'ck+':
        y_pred_ = model.predict(data, verbose=0)
        y_pred = np.argmax(y_pred_, axis=1)
        core.plot_confusion_matrix(lables, y_pred, classes=classe, normalize=True)
    else:
        return

if __name__ == "__main__":
    main(dataset='fer2013')
    # main(dataset='ck+', plot=True, cm=True)
