import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import functools
import math
from enum import Enum

def manage_logging(level=logging.INFO):
    """Decorator to set logging level at the beginning of a function and reset it at the end."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Store the original logging level
            original_level = logging.getLogger().level
            
            # Set logging level to reduce technical output from plotting libraries
            logging.getLogger().setLevel(level)
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore the original logging level
                logging.getLogger().setLevel(original_level)
            return result
        return wrapper
    return decorator

class PlotType(Enum):
    HISTOGRAM = "histogram"
    SCATTER = "scatter"

class Gryffindor_plots:
    def __init__(self, n_columns, target_column):
        # sets color palette for maximum of colors set to columns count
        self.target_column = target_column
        self.colors = sns.color_palette("viridis", n_columns)
        sns.set_theme(style="darkgrid")

    @manage_logging(logging.INFO)  # Apply the decorator
    def plots_for_columns(self, df: pd.DataFrame, columns: list, plot_type: PlotType):
        """Generate different types of plots for the given columns."""
        max_cols_per_row = 4  # Maximum number of plots per row
        ncols = min(max_cols_per_row, len(columns))  # Set columns, but not more than 4
        nrows = math.ceil(len(columns) / ncols)  # Calculate the required number of rows
        
        # Adjust figure size based on the number of rows
        fig_size = (20, 5 * nrows)  # Each row takes up approximately 5 units of height
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)
        axes = axes.flatten() if len(columns) > 1 else [axes]  # Handle case when only 1 column is provided

        for i, col in enumerate(columns):
            # label each plot with column name
            axes[i].set_title(col)
            
            # switching by plot type here
            match plot_type:
                case PlotType.HISTOGRAM:
                    sns.histplot(data=df, x=col, kde=True, bins=40, ax=axes[i], color=self.colors[i])
                case PlotType.SCATTER:
                    if self.target_column not in df.columns:
                        raise ValueError("Scatter plot requires a 'price' column in the DataFrame.")
                    sns.scatterplot(x=df[col], y=df["price"], ax=axes[i], color=self.colors[i])
                case _:
                    raise ValueError(f"Unsupported plot type: {plot_type}")
                
        # Hide any unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])  

        plt.tight_layout()
        plt.show()