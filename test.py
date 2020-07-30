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

# Splitting Data
housing = downloader.loading_housing_data()
train_set, test_set = split_train_test(housing, 0.2)

# Creatring Variuables
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# Cleaning Data
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()
housing.dropna(subset=["total_bedrooms"])
