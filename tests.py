import random as rd
import pandas as pd
import pickle
import os
import time


from data_import import read_data
from functions import clamp
from preferences import *


label_dict = {
    'clean': 0,
    'anomaly': 1,
    'attack': 2
}


class Data:

    savedir = 'src/bin'
    
    def __init__ (self, data_dict: dict):
        # Generate unique name
        self.name = f'data-{int(time.time())}'

        self.labels = list(data_dict.keys())
        self.data_dict = data_dict

        # self.features ?

    # def __repr__ (self):
    #     pass

    @classmethod
    def from_folder (cls, dirpath, labels, **kwargs):
        """
        parse folder
        :param labels: List of label_str, or dictionary {label_str => label_val}
        :param features: List of selected features
        """

        features = kwargs.get('features', None)
        filetype = kwargs.get('filetype', 'csv')

        # Read labels from folder
        data_dict = dict(( (label, dict()) for label in labels ))

        for label_str in labels:
            print(label_str)
            data_path = os.path.join(dirpath, label_str)

            for file in os.listdir(data_path):
                filepath = os.path.join(data_path, file)
                filename, ext = os.path.splitext(file)
                if ext.lower()[1:] != filetype.lower(): continue

                # Read data
                df = pd.read_csv(filepath)
                if features: df = df[features]

                # Add to dictionary
                data_dict[label_str][filename] = df

        return cls (data_dict=data_dict)

    @classmethod
    def from_pickle (cls, filepath: str):
        with open(filepath, 'rb') as archive:
            instance = pickle.load(archive)
        return instance

    def to_pickle (self, filepath: str = None):
        if not filepath:
            filepath = os.path.join(Data.savedir, f'{self.name}.bin')

        try: os.makedirs(Data.savedir)
        except FileExistsError: pass
        
        with open(filepath, 'wb') as archive:
            pickle.dump(self, archive)
        return filepath

    def split_train_test (self, train_proportion: float = 0.1, **kwargs):
        """
        Create train and test sets from data dictionary (train-centered).
        :param data_dict: Data dictionary {label:int => list(dataframe)}
        :param train_proportion: Proportion of train samples
        :param shuffle: Shuffle data before splitting
        :return: data_train_dict, data_test_dict
        """

        shuffle = kwargs.get('shuffle', False)

        self.data_train_dict, self.data_test_dict = dict(), dict()
        for label in self.labels:

            # Read data
            data = self.data_dict[label]
            data_len = len(data)

            # Calculate train and test set lengths
            data_train_len = int(data_len * train_proportion)
            data_train_len = clamp (data_train_len, 1, data_len-1)

            # Create train and test sets
            if shuffle: rd.shuffle(data)
            #TODO
            ##self.data_train_dict[label] = data[:data_train_len]
            ##self.data_test_dict [label] = data[data_train_len:]




data = Data.from_folder('src/data-2', labels=label_dict, features=SELECTED_FEATURES)
for key in data.data_dict:
    print(key)
    print(data.data_dict[key].__len__())
data.split_train_test()
