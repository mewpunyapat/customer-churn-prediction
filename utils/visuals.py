import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_numerical_distribution(df: pd.DataFrame, column: str, title: str):
    """
    Plots the distribution of a numerical column using a histogram with KDE (Kernel Density Estimation).
    
    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        column (str): Column name to plot.
        title (str): Plot title.

    Returns:
        None. Displays the plot.
    """
    plt.figure(figsize=(7, 4.5))
    sns.histplot(data=df, x=column, kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_categorical_distribution(df: pd.DataFrame, column: str, show_labels: bool):
    """
    Plots the distribution of a categorical column using a bar chart.

    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        column (str): Column name to plot.
        show_labels (bool): Whether to show count labels above bars.

    Returns:
        None. Displays the plot.
    """
    plt.figure(figsize=(7, 4.5))
    order = df[column].value_counts().index
    ax = sns.countplot(data=df, x=column, hue=column, order=order,
                       palette='pastel', edgecolor='black', legend=False)
    plt.title(f'Count of {column}', fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)

    if show_labels:
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom', fontsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def churn_rate_by_category(df: pd.DataFrame, column: str, show_labels: bool = True):
    """
    Plots average churn rate for each category in a specified categorical column.

    Args:
        df (pd.DataFrame): Input pandas DataFrame. Must include a 'Churn' column (binary: 0/1).
        column (str): Categorical column to group by.
        show_labels (bool): Whether to show churn rate labels above bars.

    Returns:
        None. Displays the plot.
    """
    churn_rates = df.groupby(column)['Churn'].mean().sort_values(ascending=False)

    plt.figure(figsize=(7, 4.5))
    ax = churn_rates.plot(kind='bar', color='salmon', edgecolor='black')
    plt.title(f'Churn Rate by {column}', fontsize=14, fontweight='bold')
    plt.ylabel('Churn Rate', fontsize=12)
    plt.xlabel(column, fontsize=12)
    plt.xticks(rotation=45)

    if show_labels:
        for i, v in enumerate(churn_rates):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
