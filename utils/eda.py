import pandas as pd
import numpy as np
import logging

def get_numerical_columns(df: pd.DataFrame) -> list:
    return df.describe().columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> list:
    return df.select_dtypes(exclude="number").columns.tolist()

def unique_values(df: pd.DataFrame, columns: list = None) -> dict:
    result = dict()
    if not columns:
        # for entire dataset
        columns = df.columns
    
    # by default - for a subset specified by a column list
    for col in columns:
        result[col] = df[col].nunique()
            
    return result

def count_rows(df: pd.DataFrame):
    return df.shape[0]

def count_cols(df: pd.DataFrame):
    return df.shape[1]

def descriptive_numerical(df: pd.DataFrame):
    pass

def descriptive_categorical(df: pd.DataFrame):
    pass

def print_eda_report(df: pd.DataFrame):
    # TODO: save the report as a text file automatically.
    numerical_columns = get_numerical_columns(df)
    categorical_columns = get_categorical_columns(df)

    print(f"================= Dataset =================")
    print(f"Dataset has shape {df.shape}")
    print()

    print(f"Dataset has numerical data in columns: {numerical_columns}")
    numerical_dict = unique_values(df, columns=numerical_columns)
    
    # sorting dictionary by values descending
    # lambda function for bonus points!
    numerical_dict = dict(sorted(numerical_dict.items(), key=lambda item: item[1], reverse=True))

    for key, value in numerical_dict.items():
        print(f'- Column "{key}" has {value} unique values.')
        if value <= 20:
            print(f"   -- Unique values are:\n {df[key].unique()}")

    print()
    
    # print("Descriptive statistics for numerical data:")
    # descriptive_numerical()

    # TODO: code duplication
    print(f"Dataset has categorical data in columns: {categorical_columns}")
    categorical_dict = unique_values(df, columns=categorical_columns)

    for key, value in categorical_dict.items():
        print(f'- Column "{key}" has {value} unique values.')
        if value <= 20:
            print(f"  -- Unique values are:\n {df[key].unique()}")

    print()

    # print("Descriptive statistics for categorical data:")
    # descriptive_categorical()
# END OF PRINT EDA
    
def auto_cleanup(df: pd.DataFrame):
    logging.info(f"Number of rows before cleanup: {count_rows(df)}")
    # find and remove empty spaces
    has_empty_spaces = df.eq(" ").sum().sum() > 0

    if has_empty_spaces:
        df.replace(" ", np.nan, inplace=True)
        logging.info(f"Dataset had empty spaces in some columns. They are replaced with NaN.")
    else:
        logging.info(f"Dataset has no empty spaces.")

    # find and remove dupolicates
    has_duplicates = False if int(df.duplicated().sum()) == 0 else True
    
    if has_duplicates:
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        logging.info(f"Dataset had duplicates. They had been dropped and index was reset.")
    else:
        logging.info(f"Dataset has no duplicates.")

    logging.info(f"Number of rows after cleanup: {count_rows(df)}")

def count_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df_nulls = pd.concat([df.isna().sum(), df.notna().sum()], ignore_index=True, axis=1)
    df_nulls.columns=['is_na', 'not_na']
    df_nulls["na_percent"] = (df_nulls["is_na"] / count_rows(df) * 100)
    # MAP for bonus points
    df_nulls["na_percent_pretty"] = df_nulls["na_percent"].map("{:.2f}%".format)
    df_nulls.sort_values(by=["na_percent", "is_na"], ascending=[False, False], inplace=True)
    return df_nulls
