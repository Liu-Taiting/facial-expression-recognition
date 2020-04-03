'''
@description: 
@author: liutaiting
@lastEditors: liutaiting
@Date: 2020-04-02 23:23:50
@LastEditTime: 2020-04-03 13:35:47
'''
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from config import NUM_CLASSES, image_width, image_height, channels, BATCH_SIZE, EPOCHS
import core

def main(dataset = 'fer2013', plot = True, cm = True):
    model = core.Alexnet()
    model.summary()
    
    if dataset == 'fer2013':
        train_flow, val_flow, X_test, y_test, num_train, num_val = core.fer2013_load_data()
        classes=np.array(("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"))
        
        model.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
                    metrics=['accuracy'])

        history = model.fit_generator(train_flow,
                            steps_per_epoch=num_train / BATCH_SIZE,
                            epochs=EPOCHS,
                            verbose=2,
                            validation_data=val_flow,
                            validation_steps=num_val / BATCH_SIZE)
    else:
        return print('wrong dataset')
    
    if plot:
        core.plot_loss(history,'Fer2013 train_loss vs val_loss')
        core.plot_acc(history,title='Fer2013 train_acc vs val_acc')
    if cm:
        confusion_matrix(model, X_test, y_test, classes)
    plt.show()

def confusion_matrix(model, data, lables, classe):
    y_pred_ = model.predict(data/255., verbose=0)
    y_pred = np.argmax(y_pred_, axis=1)
    labels = np.argmax(lables, axis=1)
    core.plot_confusion_matrix(labels,y_pred,classes= classe,normalize=True)

if __name__ == "__main__":
    main(dataset='fer2013')