'''This file groups functions for data cleaning, such as 
    formatting columns to a consistent format.'''

import pandas as pd
from datetime import datetime

def snake(data_frame):
    '''
    Converts column names to snake_case (lowercase with underscores).
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame whose columns need to be formatted.

    Returns:
        - pd.DataFrame: DataFrame with column names in snake_case.
    '''

    data_frame.columns = [column.lower().replace(" ", "_") for column in data_frame.columns]

    return data_frame

def convert_to_datetime(data_frame, columns):
    '''
    Converts specified columns to datetime format.
    
    Parameters:
        - data_frame (pd.DataFrame): The input DataFrame.
        - columns (str or list): Column name or list of column names to convert to datetime.
    
    Returns:
        - pd.DataFrame: The DataFrame with specified columns converted to datetime.
    '''

    # If a single column is provided, convert it to a list
    if isinstance(columns, str):
        columns = [columns]
    
    # Loop through each specified column and convert to datetime
    for col in columns:
        if not pd.api.types.is_datetime64_any_dtype(data_frame[col]):
            data_frame[col] = pd.to_datetime(data_frame[col], errors='coerce')
    
    return data_frame

def remove_nan_rows(data_frame):
    '''
    Removes all rows contain NaN (missing) values.

    Parameters:
    -----------
        - data_frame (pd.DataFrame): The input DataFrame.
    
    Returns:
    --------
        - pd.DataFrame: The DataFrame without the rows with NaN values.
    '''

    return data_frame.dropna()