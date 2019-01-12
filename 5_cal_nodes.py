from sklearn.externals import joblib
import numpy as np


rf = joblib.load('clf.pkl')
node_count = 0
leaf_node_indices_count = 0
for decision_tree_index, decision_tree_classifier in enumerate(rf.estimators_):
    decision_tree = decision_tree_classifier.tree_

    feature = decision_tree.feature  # feature is a ndarray, feature array stores the 1024 features' indices
    # threshold = decision_tree.threshold
    node_count += decision_tree.node_count  # node_count represent the number of nodes in the tree

    leaf_node_indices = np.array(np.where(feature < 0)).reshape(-1)  # ndarray, stores indices of leaf nodes, feature's index < 0 的index
    leaf_node_indices_count += len(leaf_node_indices)  # 每棵树里 leaf_node 的总数

print(node_count)
print(leaf_node_indices_count)