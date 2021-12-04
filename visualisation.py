import matplotlib.pyplot as plt
import pandas as pd

def plot_df (df: pd.DataFrame, title=None, ylim=None):
    plt.figure(figsize=(15, 10))
    plt.plot(df.index, df['Pouls'])
    plt.plot(df.index, df['SpO2'])
    if title: plt.title(title)
    if ylim: plt.ylim(ylim)
    plt.show()
