import random as rd
import pandas as pd
import pickle
import os
import time

from operator import itemgetter


from functions import clamp
from visualisation import plot_df
from preferences import *



class Data:

    savedir = 'src/bin'
    
    def __init__ (self, data_dict: dict):
        # Generate unique name
        self.name = f'data-{int(time.time())}'

        self.labels = list(data_dict.keys())
        self.__data_dict = data_dict

        self.__data_train_dict  = None
        self.__data_test_dict   = None
        self.__data_wtrain_dict = None
        self.__data_wtest_dict  = None

        # self.features ?

    # def __repr__ (self):
    #     pass

    def __getitem__ (self, kind):
        if self.__data_wdict: return self.__data_wdict
        else: return self.__data_dict

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
            data_path = os.path.join(dirpath, label_str)

            for file in os.listdir(data_path):
                filepath = os.path.join(data_path, file)
                filename, ext = os.path.splitext(file)
                if ext.lower()[1:] != filetype.lower(): continue

                # Read data
                df = pd.read_csv(filepath)
                if features: df = df[features]


                # TODO: need to interpolate / fill NaN values

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


    def preprocess (self):

        # Interpolate missing datapoints (NaN)
        for label_str in self.labels: #3
            datapoints = list(self.__data_dict[label_str].keys())
            
            for datapoint in datapoints:

                a = self.__data_dict[label_str][datapoint]
                a['Pouls'].interpolate(method='slinear', inplace=True)
                a['SpO2'].interpolate(method='slinear', inplace=True)



    def split_train_test (self, train_proportion: float = 0.1, **kwargs):  # needs to return two Data objects: one for train, one for test
        """
        Create train and test sets from data dictionary (train-centered).
        :param data_dict: Data dictionary {label:int => list(dataframe)}
        :param train_proportion: Proportion of train samples
        :param shuffle: Shuffle data before splitting
        :return: data_train_dict, data_test_dict
        """

        shuffle = kwargs.get('shuffle', False)

        data_train_dict, data_test_dict = dict(), dict()
        for label_str in self.labels:

            # Read data
            datapoints = list(self.__data_dict[label_str].keys())
            data_len = len(datapoints)

            # Calculate train and test set lengths
            data_train_len = int(data_len * train_proportion)
            data_train_len = clamp (data_train_len, 1, data_len-1)

            # Create train and test sets
            if shuffle: rd.shuffle(datapoints)

            datapoints_train = datapoints[:data_train_len]
            datapoints_test  = datapoints[data_train_len:]
            data_train_dict[label_str] = dict(( (label, self.__data_dict[label_str][label]) for label in datapoints_train))
            data_test_dict[label_str]  = dict(( (label, self.__data_dict[label_str][label]) for label in datapoints_test))

        # return {'train': Data(data_dict=data_train_dict), 'test': Data(data_dict=data_test_dict)}
        return Data(data_dict=data_train_dict), Data(data_dict=data_test_dict)


    def make_windows(self, window_size: int):

        self.__data_wdict = dict()
        for label_str in self.labels:

            # Init windowed dictionaries
            self.__data_wdict[label_str] = dict()

            # Get datapoints
            datapoints = list(self.__data_dict[label_str].keys())

            for datapoint in datapoints:
                df = self.__data_dict[label_str][datapoint]

                window_start = 0
                wid = 0
                while window_start + window_size <= df.shape[0]:
                    self.__data_wdict[label_str][f'{datapoint}-{wid}'] = df[window_start:window_start+window_size]
                    window_start += window_size
                    wid += 1




# data = Data.from_folder('src/data-2', labels=label_dict, features=SELECTED_FEATURES)
# data.split_train_test()
# print(data['train']['clean'].__len__())
# data.make_windows(window_size=100)
# print(data['train']['clean'].__len__())




# clustering sur les coefs de Fourier?
# regarder les coefs de la PCA
# wavelet packets

# sliding average of pulse