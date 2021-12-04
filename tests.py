import matplotlib.pyplot as plt
import numpy as np

from data import Data

from preferences import *

data = Data.from_folder('src/data-2', labels=label_dict, features=SELECTED_FEATURES)
data.split_train_test()

PRINT_DATA_LEN = False
if PRINT_DATA_LEN:
    for key in data['all']:
        print(key)
        print(data['all'][key].__len__())

PLOT_DATA = False
if PLOT_DATA:
    plt.figure(figsize=(15, 10))
    data_sel = data['all']['clean']
    for data_key in list(data_sel.keys()):
        df = data_sel[data_key]
        plt.title('clean')
        plt.plot(df.index, df['Pouls'], c='blue')
        plt.plot(df.index, df['SpO2'], c='green')
        # plot_df(df, title=f'clean - {data_key}', ylim=(60, 200))
    plt.show()

    plt.figure(figsize=(15, 10))
    data_sel = data['all']['anomaly']
    for data_key in list(data_sel.keys()):
        df = data_sel[data_key]
        plt.title('anomaly')
        plt.plot(df.index, df['Pouls'], c='blue')
        plt.plot(df.index, df['SpO2'], c='green')
        # plot_df(df, title=f'attack - {data_key}', ylim=(60, 200))
    plt.show()

    plt.figure(figsize=(15, 10))
    data_sel = data['all']['attack']
    for data_key in list(data_sel.keys()):
        df = data_sel[data_key]
        plt.title('attack')
        plt.plot(df.index, df['Pouls'], c='blue')
        plt.plot(df.index, df['SpO2'], c='green')
        # plot_df(df, title=f'attack - {data_key}', ylim=(60, 200))
    plt.show()



PLOT_FOURIER = False
if PLOT_FOURIER:

    from numpy.fft import rfft, rfftfreq

    data_sel = data['all']['attack']
    for data_key in list(data_sel.keys()):
        df = data_sel[data_key]

        # yf = scipy.fftpack.fft(df['Pouls'])
        yf = rfft(df['Pouls'])
        xf = rfftfreq(df['Pouls'].to_numpy().shape[0])
        plt.figure(figsize=(15, 10))

        plt.plot(xf[100:], yf[100:])
        plt.show()
        # break



from sklearn.ensemble import RandomForestClassifier

print('CLEAN------------------')

data_sel = data['all']['clean']

X_clean = np.stack(( data_sel[data_key] [['Pouls', 'SpO2']].to_numpy() for data_key in list(data_sel.keys()) ))
y_clean = np.zeros (X_clean.shape[0], dtype=np.int)

print('ANOMALY------------------')
data_sel = data['all']['anomaly']
X_anomaly = np.array([ data_sel[data_key] [['Pouls', 'SpO2']].to_numpy() for data_key in list(data_sel.keys()) ])
y_anomaly = np.ones (X_anomaly.shape[0], dtype=np.int)

print('ATTACK------------------')
data_sel = data['all']['attack']
X_attack = np.array([ data_sel[data_key] [['Pouls', 'SpO2']].to_numpy() for data_key in list(data_sel.keys()) ])
y_attack = np.ones (X_attack.shape[0], dtype=np.int) * 2

X = np.stack((X_clean, X_anomaly, X_attack))
y = np.stack((y_clean, y_anomaly, y_attack))

random_forest = RandomForestClassifier(n_estimators=100, max_depth=10)
random_forest.fit(X, y)

