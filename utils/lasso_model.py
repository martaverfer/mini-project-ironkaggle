# 📚 Basic libraries
import pandas as pd
import numpy as np 
import os

# 🤖 Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler


def regression_metrics(X, y, df, random_state, model, standardize=False):
    """
    Returns a dataframe with the metrics of the model you have passed as a parameter for different test sizes.
    """
    test_sizes = [0.1, 0.2, 0.3, 0.4]
    
    # List to store results for each test size
    results = []

    for i in test_sizes:
        print(f"{i*100}% test size")
        print("====================================")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=random_state)
        
        print(f'100% of our data: {len(df)}.')
        print(f'{(1-i)*100}% for training data: {len(X_train)}.')
        print(f'{i*100}% for test data: {len(X_test)}.')
        print("====================================")
        print()

        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            pass

        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Compute metrics
        r2 = r2_score(y_test, predictions)
        MSE = mean_squared_error(y_test, predictions)
        RMSE = root_mean_squared_error(y_test, predictions)
        MAE = mean_absolute_error(y_test, predictions)

        # Store the results for this test size
        results.append({
            'test_size': f'{i*100}%',
            'r2': round(r2, 3),
            'rmse': round(RMSE, 3),
            'mse': round(MSE, 3),
            'mae': round(MAE, 3)
        })

    pd.set_option('display.float_format', '{:.3f}'.format)

    # Return the final dataframe containing all results
    return pd.DataFrame(results)

def regression_validation(y_test, predictions):
    """
    Calculates R2, RMSE, MSE and MAE for a regression model and returns a dataframe
    """
    # Metric calculations
    r2 = r2_score(y_test, predictions)
    RMSE = root_mean_squared_error(y_test, predictions)
    MSE = mean_squared_error(y_test, predictions)
    MAE = mean_absolute_error(y_test, predictions)

    # Create a dataframe with the metrics
    metrics_df = pd.DataFrame({
        'Metrics': ['R2', 'RMSE', 'MSE', 'MAE'],
        'Values': [r2, RMSE, MSE, MAE],
    })

    # Set pandas display option to prevent scientific notation
    pd.set_option('display.float_format', '{:.4f}'.format)

    return metrics_df

def test_train_r_analysis(model, X_train, X_test, y_train, y_test):
    """
    Calculate R2 for training and testing sets
    """
    # Predict on the training and testing sets
    prediction_train = model.predict(X_train)
    prediction_test = model.predict(X_test)
        
    # Calculate R² scores for training and testing
    train_r2= r2_score(y_train, prediction_train)
    test_r2= r2_score(y_test, prediction_test)

    print("R2 train: ", round(train_r2, 3))
    print("R2 test: ", round(test_r2, 3))

def reporting_dataframe(X_test, y_test, model):
    """
    Returns a dataframe with actual vs. predictions 
    Evaluates and compares the predicted values of a model with the true values of the dataset
    """

    # Make predictions
    predictions = model.predict(X_test)
    
    # Create a dataframe to compare
    eval_df = pd.DataFrame({"actual": y_test, "pred": predictions})
    eval_df["dif"] = abs(eval_df["actual"]-eval_df["pred"])
    eval_df.reset_index(drop=True, inplace=True)
    
    # Return the final dataframe containing actual vs predictions
    return eval_df

def save_dataframe_to_pickle(df, filename):
    """
    Saves the DataFrame to a pickle file if the file doesn't already exist.
    """
    if not os.path.exists(filename):  # Check if the file already exists
        df.to_pickle(filename)  # Save DataFrame as pickle file
        print(f"DataFrame saved as {filename}")
    else:
        print(f"{filename} already exists. File not overwritten.")
