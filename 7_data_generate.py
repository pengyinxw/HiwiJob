from sklearn.externals import joblib
from rf_wrapper import RFWrapper
import numpy as np


rf = joblib.load('clf.pkl')
rf_wrapper = RFWrapper(rf)


def feature_block():
    x_new = rf_wrapper.generate_data(0, 0.1)

    for i in range (1,5):
        x_class = rf_wrapper.generate_data(i, 0.1)
        x_new = np.vstack((x_new, x_class))
    return x_new


new_feature = feature_block()
for j in range(9999):
    block = feature_block()
    new_feature = np.vstack((new_feature, block))

new_label = rf.predict(new_feature)

np.save('generated_feature.npy', new_feature)
np.save('generated_label.npy', new_label)
