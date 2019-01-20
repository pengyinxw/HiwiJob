#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
# from distribution_functions import sample_beta_values
from sklearn.preprocessing import binarize

class RFWrapper:

    def __init__(self, clf):
        self.clf = clf
        self.n_estimators = clf.n_estimators
        self.n_classes = clf.n_classes_

        self.n_features = 4096
        self.features_min = 0
        self.features_max = 15

        self.backward_maps = []
        self.total_count_leaf_node = [] # 每棵树里叶子结点的数目 50维
        self.total_leaf_indices = [] # 全部叶子节点的indices
        self.total_cached_backward_paths = []

        for decision_tree_index, decision_tree_classifier in enumerate(clf.estimators_):
            decision_tree = decision_tree_classifier.tree_

            feature = decision_tree.feature # feature is a ndarray, feature array stores the 1024 features' indices
            #threshold = decision_tree.threshold
            node_count = decision_tree.node_count # node_count represent the number of nodes in the tree

            #split_node_indices = np.where(feature >= 0)
            leaf_node_indices = np.array(np.where(feature < 0)).reshape(-1) # ndarray, stores indices of leaf nodes, feature's index < 0 的index
            leaf_node_indices_count = len(leaf_node_indices) # 每棵树里 leaf_node 的总数
            self.total_count_leaf_node.append(leaf_node_indices_count) # list，in which every element represent the number of leafs in a tree
            self.total_leaf_indices.append(leaf_node_indices) # list，in which every element is a ndarray, reprsent indices of leaf nodes in a tree

            backward_map = []

            for index in range(node_count):
                i = np.where(decision_tree.children_left == index) # child_left/right stores the indices of nodes
                j = np.where(decision_tree.children_right == index) # i and j are tuples
                i = np.array(i).reshape(-1)
                j = np.array(j).reshape(-1)
                if i.size > 0:
                    parent_index = i[0]
                    direction = 'l'
                elif j.size > 0:
                    parent_index = j[0]
                    direction = 'r'
                else:
                    parent_index = -1
                    direction = ''

                backward_map.append((parent_index, direction)) # list

            self.backward_maps.append(backward_map)

            cached_backward_paths = []

            for index in range(leaf_node_indices_count):
                leaf_node_index = leaf_node_indices[index]
                node_path = self.get_node_path(backward_map, leaf_node_index)
                cached_backward_paths.append(node_path)

            self.total_cached_backward_paths.append(cached_backward_paths) # list, paths from leaf node to root node

    def calc_prob(self, tree, node_index, data):
        feature = tree.feature[node_index]
        threshold = tree.threshold[node_index]

        if feature < 0:
            # print "active leaf node " + str(node_index)
            value = tree.value[node_index]
            # n_node_samples = tree.n_node_samples[node_index]

            value_rel = value / value.sum()
            return value_rel

        data_value = data[feature]
        if data_value < threshold:
            next_node_index = tree.children_left[node_index]
        else:
            next_node_index = tree.children_right[node_index]

        return self.calc_prob(tree, next_node_index, data)

    def get_node_path(self, backward_map, node_index):
        (parent_index, direction) = backward_map[node_index]

        if parent_index < 0:
            return []

        path = self.get_node_path(backward_map, parent_index)
        path.append((parent_index, direction))
        return path

    def generate_data(self, label_index, noise_prob):
        # type: (int, float) -> np.ndarray

        data = np.random.rand(self.n_features) * (self.features_max - self.features_min) + self.features_min # data is a (1024,) array
        # data_zero = np.random.randint(2, size=self.n_features)
        data_zero = np.random.uniform(size = self.n_features) # (1024,) ndarray, 1024 samples from a uniform distribution over [0,1)
        threshold = np.random.uniform(size = 1) # (1,) arrary

        data[data_zero < threshold] = 0 # modify generated data

        # data = np.zeros(self.n_features)

        data_set = np.zeros(self.n_features)
        data_set_total = np.zeros(self.n_features)

        tree_indices = list(range(0, self.n_estimators)) # a list 0~49

        random.shuffle(tree_indices) # sort all elements stochastically

        for tree_index in tree_indices:
            # tree_index = random.randint(0, n_estimators - 1)
            decision_tree_classifier = self.clf.estimators_[tree_index]
            decision_tree = decision_tree_classifier.tree_

            feature = decision_tree.feature
            threshold = decision_tree.threshold
            # node_count = decision_tree.node_count
            # backward_map = self.backward_maps[tree_index]
            leaf_node_indices = self.total_leaf_indices[tree_index] # list，in which every element is a ndarray, reprsent indices of leaf nodes in a tree
            cached_backward_paths = self.total_cached_backward_paths[tree_index] # list, paths from leaf node to root node

            sub_values = decision_tree.value[leaf_node_indices, 0, label_index]  # type: np.ndarray
            sub_values = sub_values / sub_values.sum()
            sub_values_noisy = sub_values + 1.0 * noise_prob / len(leaf_node_indices)
            sub_values_noisy = sub_values_noisy / sub_values_noisy.sum()

            count_leaf_node = self.total_count_leaf_node[tree_index]
            leaf_node_array_index = np.random.choice(count_leaf_node, p=sub_values_noisy)
            # leaf_node_array_index = random.randint(0, count_leaf_node - 1)
            # leaf_node_index = leaf_node_indices[leaf_node_array_index]

            # node_path2 = self.get_node_path(backward_map, leaf_node_index)
            node_path = cached_backward_paths[leaf_node_array_index]

            for path_node_index, direction in node_path:
                node_feature = feature[path_node_index]
                node_threshold = threshold[path_node_index]

                value = data[node_feature]
                satisfied = False

                if direction == 'l':
                    if value < node_threshold:
                        satisfied = False

                    f_min = self.features_min
                    f_max = node_threshold
                    # direction_left = True

                else:
                    if value >= node_threshold:
                        satisfied = True

                    f_min = node_threshold
                    f_max = self.features_max
                    # direction_left = False

                if not satisfied:
                   # if bordered_data:
                      #  width_value = f_max - f_min

                      #  mu_value = width_value / 4.0
                      #  std_value = width_value / 5.0

                      #  sample_value = sample_beta_values(mu_value, std_value, width_value, 1)

                      #  if direction_left:
                            # nf = f_max - sample_value
                      #  else:
                      #      nf = f_min + sample_value

                   # else:
                    nf = random.uniform(f_min, f_max)

                    data[node_feature] = nf

                    # if data_set[node_feature] > 0:
                       # print "feature already set " + str(node_feature)

                    data_set[node_feature] += 1
                    data_set_total[node_feature] = 1

        aa = data_set.sum()
        bb = data_set_total.sum()

        # prob = calc_prob(decision_tree, 0, data)
        # print prob
        return data
