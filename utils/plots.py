import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def gryffindor_histograms(columns: list, df: pd.DataFrame):
    logging.getLogger().setLevel(logging.INFO)

    # TODO: decoration of all plots to run once
    colors = sns.color_palette("viridis", len(columns))
    sns.set_theme(style="darkgrid")

    # TODO: calculate from column list length
    nrows, ncols = 2, 4  # adjust for your number of features
    
    #  FIgure size also to be calculated
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        # KDE - Kernel Density Estimation (Probability Density Function, PDF)
        sns.histplot(data=df[[col]], x=col, kde=True, bins=40, ax=axes[i], color=colors[i])
        axes[i].set_title(col)    

    plt.tight_layout()
    plt.show()

    logging.getLogger().setLevel(logging.DEBUG)