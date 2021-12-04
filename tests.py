import random as rd

from data_import import read_data
from functions import clamp


TRAIN_PROPORTION = 0.1  # train on 10% of data




class Data:
    pass


def create_train_test_sets (data_dict: dict, train_proportion: float = 0.1, **kwargs):
    """
    Create train and test sets from data dictionary (train-centered).
    :param data_dict: Data dictionary {label:int => list(dataframe)}
    :param train_proportion: Proportion of train samples
    :param shuffle: Shuffle data before splitting
    :return: data_train_dict, data_test_dict
    """

    shuffle = kwargs.get('shuffle', False)

    data_train_dict, data_test_dict = dict(), dict()
    for label in data_dict.keys():

        # Read data
        data = data_dict[label]
        data_len = len(data)

        # Calculate train and test set lengths
        data_train_len = int(data_len * train_proportion)
        data_train_len = clamp (data_train_len, 1, data_len-1)

        # Create train and test sets
        if shuffle: rd.shuffle(data)
        data_train_dict[label] = data[:data_train_len]
        data_test_dict [label] = data[data_train_len:]
    
    return data_train_dict, data_test_dict



def create_windows (vals, window_len):
    pass



# Create dictionary of data
data_dict = read_data()

# Create train and test sets
data_train_dict, data_test_dict = create_train_test_sets (data_dict)
