import matplotlib.pyplot as plt
import pandas as pd

def plot_df (df: pd.DataFrame):
    plt.figure(figsize=(15, 10))
    plt.plot(df.index, df['Pouls'])
    plt.plot(df.index, df['SpO2'])
    plt.show()
