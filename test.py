"""
Author: Aleksa Zatezalo
Date: July 30, 2020
Description: A script used to partition test data.
"""

import numpy as np
import downloader

def split_train_test(data, test_ratio):
    """
    Takes variables data(an array) and test_ratio a float.
    Based on the proportion dictated in test_ratio it splits
    the data into a training and testing partition.
    """

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

housing = downloader.loading_housing_data()
train_set, test_set = split_train_test(housing, 0.2)

