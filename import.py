import pandas as pd

import os

from visualisation import plot_df
from preferences import *

SRC_PATH = 'src'


class ImportManager:

    def __init__ (self, data_folder='data-2'):
        self.df_clean   = None
        self.df_anomaly = None
        self.df_attack  = None

    def import_ (self, data_types: dict):
        pass

data_types = {
    'clean': 0,
    'anomaly': 1,
    'attack': 2
}

# data_types = {'attack': 2}
# data_types = {'anomaly': 1}
data_types = {'clean': 0}

data_list = list()
data_path = os.path.join(SRC_PATH, 'data-2')

for data_type in data_types.keys():
    dirpath = os.path.join(data_path, data_type)

    for file in os.listdir(dirpath):
        filepath = os.path.join(dirpath, file)
        df = pd.read_csv(filepath)


        df = df[SELECTED_FEATURES]
        plot_df(df)

        # Add label column
        label_val = data_types[data_type]
        label = pd.DataFrame(label_val, index=df.index, columns=['label'])
        df = pd.concat((df, label), axis=1)

        print(df)
