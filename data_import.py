import pandas as pd
import os

from visualisation import plot_df
from preferences import *


data_types = {
    'clean': 0,
    'anomaly': 1,
    'attack': 2
}


def read_data():

    data_dict = {
        'clean': list(),
        'anomaly': list(),
        'attack': list()
    }

    data_path = os.path.join(SRC_PATH, 'data-2')

    for data_type in data_types.keys():
        dirpath = os.path.join(data_path, data_type)

        for file in os.listdir(dirpath):
            filepath = os.path.join(dirpath, file)
            df = pd.read_csv(filepath)

            # Select features
            df = df[SELECTED_FEATURES]

            data_dict[data_type].append(df)

    return data_dict


data_dict = read_data()
