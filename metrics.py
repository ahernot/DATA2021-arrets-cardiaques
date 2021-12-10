# Analyse signal

import numpy as np
import pandas as pd



def get_metrics (sig_window: pd.DataFrame):
    pulse = sig_window['Pouls']
    pulse_std = np.std(pulse)
    
    #pulse max derivative?
    #correlation b/w pulse and SpO2

    


"""
then data.apply_metrics(metrics_func)
and will result in data.X containing metrics
and also data.__metrics_dict
"""