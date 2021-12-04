
def clamp (val, start, stop):
    """
    Clamp value in [start, stop]
    """
    if val < start:  return start
    elif val > stop: return stop
    else:            return val
