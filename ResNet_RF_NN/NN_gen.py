import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"        # only use gpu0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import regularizers
from sklearn.externals import joblib
from rf_wrapper import RFWrapper

rf = joblib.load('clf.pkl')
rf_wrapper = RFWrapper(rf)

# load validation data
x_val = np.load('val_x.npy')
y_val = np.load('val_y.npy')
y_val = np_utils.to_categorical(y_val)
validation_data = (x_val, y_val)


# define a generator
def data_generator(batch_size):
    batch_input = []
    batch_output = []
    labels = np.random.choice(a = 5, size = batch_size)

    for label in labels:
        input = rf_wrapper.generate_data(label, 0.1)
        input = input.reshape(1,-1)
        output = rf.predict(input)

        batch_input += [input]
        batch_output += [output]

    batch_x = np.array(batch_input)
    batch_x = batch_x.reshape(-1, batch_x.shape[2])
    batch_y = np.array(batch_output)
    batch_y = np_utils.to_categorical(batch_y)

    yield(batch_x, batch_y)


train_generator = data_generator(32)

# new a 2_hidden layers Net
model = Sequential()
model.add(Dense(input_dim = 4096, units = 128, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
#model.add(Dense(units = 512,activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(units = 256,activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(units = 5, activation = 'softmax'))
print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=0.001, decay=1e-6), metrics = ['accuracy'])
train_history = model.fit_generator(train_generator, epochs = 50, validation_data = validation_data)

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
