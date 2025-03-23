import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import functools
import math
from enum import Enum

# Global config (default values, can be overridden)
TARGET_COLUMN = None
COLORS = None
MAX_COLUMNS = 4

class PlotType(Enum):
    HISTOGRAM = "histogram"
    SCATTER = "scatter"

def manage_logging(level=logging.INFO):
    """
    Decorator to set logging level at the beginning of a function and reset it at the end.
    This helps to truncate the output from technical information from visual libraries when making plots.
    """
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

def plot_config(target_column, max_colors, max_columns=4):
    """Set global config for plotting."""

    global TARGET_COLUMN, COLORS, MAX_COLUMNS

    TARGET_COLUMN = target_column
    COLORS = sns.color_palette("viridis", max_colors)
    MAX_COLUMNS = max_columns

    sns.set_theme(style="darkgrid")

@manage_logging(logging.INFO)  # Apply decorator to reset logging level
def plots_for_numeric_columns(df: pd.DataFrame, columns: list, plot_type: PlotType, plot_title: str):
    """Generate different types of plots for the given columns."""

    ncols = min(MAX_COLUMNS, len(columns))  # Set columns, but not more than MAX_COLUMNS
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
                sns.histplot(data=df, x=col, kde=True, bins=40, ax=axes[i], color=COLORS[i])
            case PlotType.SCATTER:
                if TARGET_COLUMN not in df.columns:
                    raise ValueError("Scatter plot requires a '{TARGET_COLUMN}' column in the DataFrame.")
                sns.scatterplot(x=df[col], y=df["price"], ax=axes[i], color=COLORS[i])
            case _:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  
   
    # Add a big header for the whole figure
    fig.suptitle(plot_title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()