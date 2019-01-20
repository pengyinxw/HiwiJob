# split cifar10 into two subsets

from keras.datasets import cifar10
import numpy as np
np.random.seed(10)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('train:', 'images:', x_train.shape,
                'labels:', y_train.shape)
print('test:',  'images:', x_test.shape,
                'labels:', y_test.shape)

y_label = y_train.reshape(-1)


# first subset with 5 classes

cifarA_indices = np.array(np.where(y_label < 5)).reshape(-1)
cifarA_x_train = x_train[cifarA_indices]
cifarA_y_train = y_train[cifarA_indices]
print(cifarA_x_train.shape)
print(cifarA_y_train.shape)
np.save('cifarA_x.npy', cifarA_x_train)
np.save('cifarA_y.npy', cifarA_y_train)


# second subset with 5 classes

cifarB_indices = np.array(np.where(y_label > 4)).reshape(-1)
cifarB_x_train = x_train[cifarB_indices]
cifarB_y_train = y_train[cifarB_indices]
cifarB_y_train = cifarB_y_train - 5
print(cifarB_x_train.shape)
print(cifarB_y_train.shape)
np.save('cifarB_x.npy', cifarB_x_train)
np.save('cifarB_y.npy', cifarB_y_train)