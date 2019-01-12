import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"        # only use gpu0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils


x_train = np.load('generated_feature.npy')
y_train = np.load('generated_label.npy')
x_train = x_train.astype('float32')
y_train = y_train - 5*np.ones([50000,])
y_train = np_utils.to_categorical(y_train)
print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Dense(input_dim = 4096, units = 128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 256,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 5, activation = 'softmax'))

print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
clf = model.fit(x_train, y_train, validation_split = 0.2, epochs = 50, batch_size = 32, verbose = 2)
