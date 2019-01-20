import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"        # only use gpu0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from keras.models import Model
import numpy as np


x_train = np.load('cifarB_x.npy')
y_train = np.load('cifarB_y.npy')

x_train = x_train.astype('float32')
x_train = preprocess_input(x_train)
y_label = y_train.reshape(-1)

# unbalanced dataset???
# c = y_train[:800]
# print((c == 9).sum())


# limit dataset to 800 examples

cifarB_indices = np.array(np.where(y_label == 0)).reshape(-1)
cifarB_x_train = x_train[cifarB_indices][:160]
cifarB_y_train = y_train[cifarB_indices][:160]

for i in range(1,5):
    cifarB_indices = np.array(np.where(y_label == i)).reshape(-1)
    cifarB_x = x_train[cifarB_indices][:160]
    cifarB_y = y_train[cifarB_indices][:160]
    cifarB_x_train = np.vstack((cifarB_x_train, cifarB_x))
    cifarB_y_train = np.vstack((cifarB_y_train, cifarB_y))

print(cifarB_x_train.shape)
print(cifarB_y_train.shape)


# randomly permute dataset
m = len(cifarB_y_train)
indices = np.random.permutation(m)
x_train = cifarB_x_train[indices]
y_train = cifarB_y_train[indices]


model = load_model('ResNet50.h5')
outputs = model.get_layer('activation_28').output
model = Model(inputs = model.input, outputs = outputs)
print(model.summary())

x_train = model.predict(x_train)
print(x_train.max())
print(x_train.min())

np.save('features_x.npy', x_train)
np.save('features_y.npy', y_train)
