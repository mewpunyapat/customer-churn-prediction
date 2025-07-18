import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_numerical_distribution(df:pd.DataFrame, column:str, title:str ):
    plt.figure(figsize=(6,4))
    sns.histplot(data=df[column], kde=True, bins=30)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
