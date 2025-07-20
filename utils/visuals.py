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

def stacked_bar_churn_ratio(df: pd.DataFrame, column: str):
    """
    Plots stacked bar chart of churn and non-churn proportions for a categorical column.

    Args:
        df (pd.DataFrame): Must include 'Churn' column.
        column (str): Categorical column to analyze.
    """
    prop_df = (df.groupby([column, 'Churn']).size() / df.groupby(column).size()).unstack().fillna(0)
    prop_df.plot(kind='bar', stacked=True, figsize=(7, 4.5), color=['lightgreen', 'salmon'], edgecolor='black')

    plt.title(f'Proportion of Churn by {column}', fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def boxplot_by_churn(df: pd.DataFrame, column: str):
    """
    Shows boxplot of a numerical column split by churn status.

    Args:
        df (pd.DataFrame): Must contain 'Churn' and the given column.
        column (str): Numerical column to compare.
    """
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x='Churn', y=column, palette='Set2')
    plt.title(f'{column} Distribution by Churn', fontsize=14, fontweight='bold')
    plt.xlabel('Churn', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def countplot_with_churn(df: pd.DataFrame, column: str):
    """
    Displays countplot of a categorical column with hue based on churn status.

    Args:
        df (pd.DataFrame): DataFrame with a 'Churn' column.
        column (str): Categorical column to analyze.
    """
    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=df, x=column, hue='Churn', palette='pastel', edgecolor='black')
    plt.title(f'Count of Customers by {column} and Churn Status', fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def kdeplot_by_churn(df: pd.DataFrame, column: str):
    """
    Plots KDE (density) plots of a numerical column for churned vs non-churned.

    Args:
        df (pd.DataFrame): Must include 'Churn' column.
        column (str): Numerical column to analyze.
    """
    plt.figure(figsize=(7, 4.5))
    for churn_value, label in zip([0, 1], ['Not Churned', 'Churned']):
        sns.kdeplot(df[df['Churn'] == churn_value][column], label=label, fill=True, alpha=0.5)

    plt.title(f'Distribution of {column} by Churn', fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
