import matplotlib.pyplot as plt
import numpy as np

from data import Data
from functions import moving_avg_kernel

from preferences import *
from visualisation import plot_df

data = Data.from_folder('src/data-2', labels=label_dict, features=SELECTED_FEATURES)


data.preprocess()

data_train, data_test = data.split_train_test()
# data.make_windows(window_size=100)

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


for key, df in data['train']['clean'].items():

    plt.figure(figsize=(15, 10))
    # plt.plot(df.index, df['Pouls'])
    
    # Average
    avg = np.mean(df['Pouls'])
    # plt.plot(df.index, np.ones_like(df.index) * avg)

    # Moving average
    avg_rad = 5
    mak = moving_avg_kernel(radius=avg_rad)
    m_avg = np.convolve(df['Pouls'].to_numpy(), mak, mode='valid')    
    plt.plot(df.index[avg_rad:-avg_rad], m_avg)

    plt.show()

    # break



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




CLASSIFIERS = False
if CLASSIFIERS:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    data_sel = data['train']['clean']
    # X_clean = np.array([ data_sel[data_key] [['Pouls', 'SpO2']].to_numpy() for data_key in list(data_sel.keys()) ])
    X_clean = np.array([ np.concatenate((data_sel[data_key]['Pouls'].to_numpy(), data_sel[data_key]['SpO2'].to_numpy())) for data_key in list(data_sel.keys()) ])
    print(X_clean.shape)
    y_clean = np.zeros (X_clean.shape[0], dtype=np.int)

    data_sel = data['train']['anomaly']
    # X_anomaly = np.array([ data_sel[data_key] [['Pouls', 'SpO2']].to_numpy() for data_key in list(data_sel.keys()) ])
    X_anomaly = np.array([ np.concatenate((data_sel[data_key]['Pouls'].to_numpy(), data_sel[data_key]['SpO2'].to_numpy())) for data_key in list(data_sel.keys()) ])
    y_anomaly = np.ones (X_anomaly.shape[0], dtype=np.int)
    print(X_anomaly.shape)


    data_sel = data['train']['attack']
    # X_attack = np.array([ data_sel[data_key] [['Pouls', 'SpO2']].to_numpy() for data_key in list(data_sel.keys()) ])
    X_attack = np.array([ np.concatenate((data_sel[data_key]['Pouls'].to_numpy(), data_sel[data_key]['SpO2'].to_numpy())) for data_key in list(data_sel.keys()) ])
    y_attack = np.ones (X_attack.shape[0], dtype=np.int) * 2
    print(X_attack.shape)

    X = np.concatenate((X_clean, X_anomaly, X_attack), axis=0)
    y = np.concatenate((y_clean, y_anomaly, y_attack), axis=0)



