'''This module contains functions and configurations for visualizing data, 
    including setting up a clean and professional styling for plots, defining 
    color palettes, and handling font customization. 
'''

# Standard Libraries
import os

# Data Handling & Computation
import pandas as pd
import numpy as np

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

# ==============================
# Plot Styling & Customization
# ==============================

# Set a Minimalist Style
sns.set_style("whitegrid")

# Customize Matplotlib settings
mpl.rcParams.update({
    'axes.edgecolor': 'grey',       
    'axes.labelcolor': 'black',     
    'xtick.color': 'black',         
    'ytick.color': 'black',         
    'text.color': 'black'           
})

# General color palette for plots
custom_colors = ["#1F4E79", "#8F2C78", '#6EE5D9', '#EFCC86', '#767474']

# Markers for assets
custom_markers = ['o', 's', '^', 'D', 'v']

# ==============================
# Font Configuration
# ==============================

# Path to the custom font file
FONT_PATH = './scp/fonts/Montserrat-Regular.ttf'

# Add the font to matplotlib's font manager
font_manager.fontManager.addfont(FONT_PATH)

# Set the font family to Montserrat
plt.rcParams['font.family'] = 'Montserrat'

# ==============================
# Custom Formatter Functions
# ==============================

def currency_formatter(x, pos):
    '''
    Custom formatter function to display y-axis values,
    formatted as currency with comma separators.

    Parameters:
    - x (float): The numerical value to format.
    - pos (int): The tick position (required for matplotlib formatters).

    Returns:
    - str: Formatted string representation of the value.
    '''
    return f'${x:,.2f}'

# ==============================
# Vizualization Functions
# ==============================

def assets_trends(data_frame):
    '''
    This function creates a time series of the different assets.
    
    Parameters:
        - data_frame (pd.DataFrame): The dataset containing datetime column and the assests.
    Returns:
        - None: It show the plots.
    '''


    fig, ax = plt.subplots(figsize=(12, 6))
    for i, column in enumerate(data_frame.columns[1:]): 
        formatted_label = column.replace('asset', 'Asset ') 
        ax.plot(data_frame['date'], 
                data_frame[column], 
                label=formatted_label, 
                marker=custom_markers[i], 
                color=custom_colors[i])
        
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    ax.set_title('Asset Performance Over Time', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()    

def daily_return(data_frame):
    '''
    The function calculates the daily percentage return for each asset.

    Parameters:
        -data_frame (pd.DataFrame): DataFrame containing the assets.

    Returns:
        - df (pd.DataFrame(): A new DataFrame containing the daily returns of each asset.
    '''

    df = pd.DataFrame()

    for asset in data_frame.columns[1:]:
        df[f'{asset}_daily_return'] = data_frame[asset].pct_change(fill_method=None) * 100

    return df

def plot_correlation_heatmap(df):
    '''
    Generates a heatmap to visualize the correlation coefficients between numerical variables.

    Parameters:
        -df (pd.DataFrame): DataFrame of the daily returns of each asset.

    Returns:
        - None: Plot and save the correlation map.
    '''

    df.dropna(inplace=True) 
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(10, 7))
    sns.heatmap(correlation_matrix, 
                cmap="magma_r", 
                linewidths=0.5, 
                annot=True, 
                fmt=".2f",
                xticklabels=[col.replace('_', ' ').replace('asset', 'asset ').title() for col in df.columns],
                yticklabels=[col.replace('_', ' ').replace('asset', 'asset ').title() for col in df.columns], 
                mask=mask
                )
    plt.title("Correlation Heatmap", fontsize=14)

def plot_scatter_plot_all(df):
    '''
    Generates a scatter plot between the returns of all the assets.

    Parameters:
        - df (pd.DataFrame):  DataFrame of the daily returns of each asset.

    Returns:
        - None: Plot and show the scatter plot.
    '''

    num_assets = df.shape[1]
    fig, axes = plt.subplots(nrows=num_assets , ncols=num_assets - 1, figsize=(14, 20))

    for i, asset1 in enumerate(df.columns[:]):
        for j, asset2 in enumerate(df.columns[:]):
            if i != j:
                if j > i:
                    ax = axes[i, j-1]
                else:
                    ax = axes[i, j]
                sns.scatterplot(x=df[asset2], y=df[asset1], ax=ax, color=custom_colors[1], alpha=0.7)
                
                ax.set_xlabel(f"{asset2.replace('_', ' ').replace('asset', 'Asset ').title()} (%)", fontsize=10)
                ax.set_ylabel(f"{asset1.replace('_', ' ').replace('asset', 'Asset ').title()} (%)", fontsize=10)

    plt.suptitle("Daily Returns Relationship", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_scatter_plot(df, asset1, asset2):
    '''
    Generates a scatter plot between the returns of two assets.

    Parameters:
        - df (pd.DataFrame):  DataFrame of the daily returns of each asset.
        - asset1 (str): The first asset's column name.
        - asset2 (str): The second asset's column name.

    Returns:
        - None: Plot and show the scatter plot.
    '''

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[asset1], y=df[asset2], color=custom_colors[0], alpha=0.6)
    plt.title(f"Daily Returns Relationship", fontsize=14)
    plt.xlabel(f"{asset1.replace('_', ' ').replace('asset', 'asset ').title()} (%)", fontsize=12)
    plt.ylabel(f"{asset2.replace('_', ' ').replace('asset', 'asset ').title()} (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_area_chart(data_frame):
    '''
    Plots an area chart of asset weights over time. 
    The chart will show the cumulative weight of each asset over time, stacked on top of each other.

    Parameters:
     -data_frame (pd.DataFrame): A pandas DataFrame with the date as the index
                                and asset weights as columns.
    
    Returns:
        - None: Plot and show the area chart.
    '''

    df = data_frame.copy()
    df.set_index('date', inplace=True)
    formated_labels = [col.replace('asset', 'asset ').title() for col in df.columns]
  
    plt.figure(figsize=(12, 6))
    plt.stackplot(df.index, df.values.T, labels=formated_labels, alpha=1.0, colors=custom_colors)
    plt.title('Assets Weights Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(fontsize=12, loc= 'upper left', bbox_to_anchor=(1.01, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_cumulative_returns(data_frame, data_frame_2):
    '''
    Plots the historical cumulative returns of the portfolio.

    Parameters:
        - data_frame (pd.DataFrame): A pandas DataFrame with the date 
                                    and portfolio values over time.
        - data_frame_2 (pd.DataFrame): A pandas DataFrame with the date 
                                    and portfolio weighhts over time.
    
    Returns:
        - cumulative_returns_p (pandas.Series): Portfolio cumulative returns
    '''
    
    df = data_frame.copy()
    df.set_index('date', inplace=True)

    df_2 = data_frame_2.copy()
    df_2.set_index('date', inplace=True)

    df_merge = df.merge(df_2, on= "date", how='left')
    df_merge.dropna(inplace=True)

    daily_returns = df_merge.iloc[:,:len(df.columns)].pct_change(fill_method=None)
    weights = df_merge.iloc[:, len(df.columns):]
    daily_returns.columns = [col.replace('_x', '') for col in daily_returns.columns]
    weights.columns = [col.replace('_y', '') for col in weights.columns]

    weighted_returns = daily_returns * weights
    portfolio_returns = weighted_returns.sum(axis=1)
    cumulative_returns = (1 + weighted_returns).cumprod() - 1
    cumulative_returns_p = (1 + portfolio_returns).cumprod() - 1

    plt.figure(figsize=(12, 6))
    for asset, color in zip(cumulative_returns.columns, custom_colors):
        plt.plot(cumulative_returns.index, cumulative_returns[asset], label=asset.replace('asset', 'asset ').title(), color=color, linewidth=2)

    plt.plot(cumulative_returns_p.index, cumulative_returns_p, label="Portfolio Return", color="black", linewidth=2)

    plt.title("Historical Cumulative Returns of the Portfolio", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Returns", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(fontsize=12, loc= 'upper left', bbox_to_anchor=(1.01, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.show()

    return cumulative_returns_p, portfolio_returns

def calculate_annualized_return(cumulative_returns_p):
    '''
    Calculate the annualized return of the portfolio.
    
    Parameters:
     - cumulative_returns_p (pd.Series): A pandas series with the date 
                                    and portfolio cumulative return.
    
    Returns:
        - annualized_portfolio_return (float): The annualized return of the portfolio.
    '''
    
    total_years = (cumulative_returns_p.index[-1] - cumulative_returns_p.index[0]).days / 261
    annualized_portfolio_return = (1 + cumulative_returns_p.iloc[-1]) ** (1 / total_years) - 1
    
    return annualized_portfolio_return

def calculate_annualized_volatility(portfolio_returns, annualization_factor=261):
    '''
    Calculates the annualized volatility of the portfolio.

    Parameters:
     - annualized_portfolio_return (pd.Series): A pandas Series with the daily portfolio returns.
     - annualization_factor (int): The number of trading days in a year (default is 261).
    
    Returns:
        - float: Annualized volatility of the portfolio.
    '''

    daily_volatility = portfolio_returns.std()
    annualized_volatility = daily_volatility * np.sqrt(annualization_factor)
    
    return annualized_volatility

def plot_category_area_chart(asset_weights, asset_info):
    '''
    Plots an area chart of asset weights over time, grouped by asset category.

    Parameters:
     - asset_weights (pd.DataFrame): A pandas DataFrame where rows are dates 
                                     and columns are asset names with their weights.
     - asset_info (pd.DataFrame): A pandas DataFrame with two columns: 'Name' (asset name)
                                  and 'Family' (category of asset).

    Returns:
        - None: Plots and shows the area chart.
    '''

    asset_info["Name"] = asset_info["Name"].str.lower()
    asset_info_dict = dict(zip(asset_info["Name"], asset_info["Family"]))

    category_weights = asset_weights.copy()
    category_weights = category_weights.rename(columns=asset_info_dict)
    
    category_weights = category_weights.T.groupby(level=0).sum().T
    category_weights['date'] = pd.to_datetime(category_weights['date'], errors='coerce')    
    category_weights.set_index('date', inplace=True)
    category_weights = category_weights.apply(pd.to_numeric)

    plt.figure(figsize=(12, 6))
    category_colors = ['#8b0000', '#ff6347', '#32cd32']
    plt.stackplot(category_weights.index, category_weights.T.values, 
                  labels=category_weights.columns, 
                  colors=category_colors,
                  alpha=0.8)
    plt.title('Asset Category Weights Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()