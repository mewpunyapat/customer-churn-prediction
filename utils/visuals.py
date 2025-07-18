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

def plot_categorical_distribution(df:pd.DataFrame, column:str):
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=column, order=df[column].value_counts().index)
    plt.title(f'Count of {column}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def churn_rate_by_category(df:pd.DataFrame, column:str):
    churn_rates = df.groupby(column)['Churn'].mean().sort_values(ascending=False)
    churn_rates.plot(kind='bar', figsize=(6,4), color='tomato')
    plt.title(f'Churn Rate by {column}')
    plt.ylabel('Churn Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()