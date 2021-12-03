import pandas as pd
import matplotlib.pyplot as plt

import os

from preferences import *


DATA_PATH = 'src/data-2'
data_types = {
    'clean': 0,
    'anomaly': 1,
    'attack': 2
}

data_types = {'attack': 2}


df_full = pd.DataFrame()
for data_type in data_types.keys():
    dirpath = os.path.join(DATA_PATH, data_type)

    for file in os.listdir(dirpath):
        filepath = os.path.join(dirpath, file)
        df = pd.read_csv(filepath)

        df_full = pd.concat((df_full, df))


df_full.reset_index(inplace=True)
        


plt.figure(figsize=(15, 10))
plt.plot(df_full.index, df_full['Pouls'])
plt.plot(df_full.index, df_full['SpO2'])
plt.show()
