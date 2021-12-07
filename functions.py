import numpy as np


def clamp (val, start, stop):
    """
    Clamp value in [start, stop]
    """
    if val < start:  return start
    elif val > stop: return stop
    else:            return val



def moving_avg_kernel (radius: int = 0):
    diameter = max(int(radius * 2 + 1), 1)
    return np.ones(diameter) / diameter
