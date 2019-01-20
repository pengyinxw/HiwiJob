from sklearn.externals import joblib
from rf_wrapper import RFWrapper
import numpy as np
from keras.utils import np_utils


rf = joblib.load('clf.pkl')
rf_wrapper = RFWrapper(rf)


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


gens = data_generator(32)


