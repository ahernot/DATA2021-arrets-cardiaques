# Analyse signal

import numpy as np
from numpy.lib.function_base import average
from scipy.signal import correlate
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing


DER_KERNEL = np.array([-1/2, 0, 1/2])
# get function to generate DER_KERNEL


def get_metrics (df: pd.DataFrame):
    hr = df['Pouls']
    spo2 = df['SpO2']

    # Get moments of hr
    hr_stdev = np.std(hr)
    hr_avg = np.average(hr)

    # Get max derivative of hr
    spo2_der = np.convolve(spo2, DER_KERNEL, mode='valid')
    hr_der = np.convolve(hr, DER_KERNEL, mode='valid')  # extra crop along edges
    hr_der_max = np.max(hr_der)

    #correlation b/w pulse and SpO2
    #   => standardise both and calculate distance between; get avg(L2) / max(L∞) distance
    #TODO: how to calculate correlation between two curves? first order correlatioin with relative evolution of standardfised derivatives?
    hr_std = preprocessing.StandardScaler().fit(hr.to_numpy().reshape(-1, 1)) .transform(hr.to_numpy().reshape(-1, 1))
    spo2_std  = preprocessing.StandardScaler().fit(spo2.to_numpy().reshape(-1, 1)) .transform(spo2.to_numpy().reshape(-1, 1))

    # plt.figure(figsize=(15, 10))
    # plt.plot(range(hr_std.__len__()), hr_std)
    # plt.plot(range(spo2_std.__len__()), spo2_std)
    # plt.show()
    
    # dist = np.abs(hr_std - spo2_std)
    # dist_score = np.linalg.norm(dist)
    

    # spectral power density? isn't this a time-averaged spectrogram basically?


    

    # print(hr.shape, spo2.shape)
    

    # TODO: cross-correlate both derivatives
    # plt.figure(figsize=(15, 10))
    # plt.plot(range(spo2_der.__len__()), spo2_der)
    # plt.plot(range(hr_der.__len__()), hr_der)
    # plt.show()

    corr = correlate(hr, spo2, mode='valid') [0]

    metrics_list = [
        hr_avg,
        hr_stdev,
        hr_der_max,
        corr
    ]


    return np.array(metrics_list)




"""
then data.apply_metrics(metrics_func)
and will result in data.X containing metrics
and also data.__metrics_dict
"""

# clustering sur les coefs de Fourier?
# regarder les coefs de la PCA
# wavelet packets

# sliding average of pulse => smoothed max derivative? or multiple datapoints?
