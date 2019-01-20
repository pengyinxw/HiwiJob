import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"        # only use gpu0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import regularizers


x_train = np.load('generated_feature.npy')
y_train = np.load('generated_label.npy')
x_train = x_train.astype('float32')
y_train = y_train
y_train = np_utils.to_categorical(y_train)
print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Dense(input_dim = 4096, units = 55, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
#model.add(Dense(units = 512,activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(units = 64, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(units = 5, activation = 'softmax'))

print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=0.001, decay=1e-6), metrics = ['accuracy'])
train_history = model.fit(x_train, y_train, validation_split = 0.2, epochs = 50, batch_size = 32, verbose = 1)


import matplotlib.pyplot as plt

def plot_train_history(train_history, acc, val_acc, loss, val_loss, ACC, LOSS):
    plt.figure()
    plt.plot(train_history.history[acc])
    plt.plot(train_history.history[val_acc])
    plt.title('Training and validation accuracy')
    plt.ylabel(acc)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig(ACC)

    plt.figure()
    plt.plot(train_history.history[loss])
    plt.plot(train_history.history[val_loss])
    plt.title('Training and validation loss')
    plt.ylabel(loss)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig(LOSS)

plot_train_history(train_history, 'acc', 'val_acc', 'loss', 'val_loss', 'acc.pdf', 'loss.pdf')
