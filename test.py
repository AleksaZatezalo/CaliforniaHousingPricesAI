"""
Author: Aleksa Zatezalo
Date: July 30, 2020
Description: A script used to partition test data.
"""

import numpy as np

def split_train_test(data, test_ratio):
    """
    Takes variables data(an array) and test_ratio a float.
    Based on the proportion dictated in test_ratio it splits
    the data into a training and testing partition.
    """

    shuffled_indicies = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indicies = shuffled_indicies[:test_set_size]
    train_indices = shuffled_indicies[test_indicies:]
    return data.iloc[train_indices], data.iloc[train_indices]

