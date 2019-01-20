import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"        # only use gpu0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np

x_train = np.load('features_x.npy')
y_train = np.load('features_y.npy')

x_train = x_train.reshape(800, -1)
#y_train = y_train.reshape(-1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.5)

clf = RandomForestClassifier(n_estimators = 100)
clf.fit(x_train, y_train)
Val_acc = clf.score(x_val, y_val)
print('val_acc= ', Val_acc)

joblib.dump(clf, 'clf.pkl')
np.save('val_x.npy', x_val)
np.save('val_y.npy', y_val)
