# Analyse signal

import numpy as np
from numpy.lib.function_base import average
import pandas as pd

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
    hr_der = np.convolve(hr, DER_KERNEL, mode='valid')  # extra crop along edges
    hr_der_max = np.max(hr_der)

    #correlation b/w pulse and SpO2
    #   => standardise both and calculate distance between; get avg(L2) / max(L∞) distance
    #TODO: how to calculate correlation between two curves? first order correlatioin with relative evolution of standardfised derivatives?
    hr_std = preprocessing.StandardScaler().fit(hr.to_numpy().reshape(-1, 1)) .transform(hr.to_numpy().reshape(-1, 1))
    spo2_std  = preprocessing.StandardScaler().fit(spo2.to_numpy().reshape(-1, 1)) .transform(spo2.to_numpy().reshape(-1, 1))
    dist = np.abs(hr_std - spo2_std)
    dist_score = np.linalg.norm(dist)
    

    # spectral power density? isn't this a time-averaged spectrogram basically?


    metrics_list = [
        hr_avg,
        hr_stdev,
        hr_der_max,
        dist_score
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
