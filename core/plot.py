'''
@description: Plot loss and accuracy images
@author: liutaiting
@lastEditors: liutaiting
@Date: 2020-04-03 10:59:58
@LastEditTime: 2020-04-03 12:05:14
'''
from matplotlib import pyplot as plt
from config import EPOCHS
import tensorflow as tf

def plot_loss(history, title = 'Title',save = False,savename = ''):
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']

    epoch = range(EPOCHS)
    plt.figure()
    plt.plot(epoch,train_loss,'r', label='train_loss')
    plt.plot(epoch,val_loss,'b', label='val_loss')
    plt.title('%s'%title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        plt.savefig("%s.png"%savename,dpi = 300)

def plot_acc(history, title = 'Title',save = False,savename = ''):
    train_loss=history.history['accuracy']
    val_loss=history.history['val_accuracy']

    epoch = range(EPOCHS)
    plt.figure()
    plt.plot(epoch,train_loss,'r', label='train_accuracy')
    plt.plot(epoch,val_loss,'b', label='val_accuracy')
    plt.title('%s'%title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        plt.savefig("%s.png"%savename,dpi = 300)