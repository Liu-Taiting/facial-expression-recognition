'''
@description: 
@author: liutaiting
@lastEditors: liutaiting
@Date: 2020-04-02 23:23:50
@LastEditTime: 2020-04-03 10:48:24
'''
import tensorflow as tf
from config import NUM_CLASSES, image_width, image_height, channels, BATCH_SIZE
import core

model = core.Alexnet()
model.summary()
train_flow, val_flow, num_train, num_val = core.fer2013_load_data()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

num_epochs = 5
history = model.fit_generator(train_flow,
                    steps_per_epoch=num_train / BATCH_SIZE,
                    epochs=num_epochs,
                    verbose=2,
                    validation_data=val_flow,
                    validation_steps=num_val / BATCH_SIZE) 