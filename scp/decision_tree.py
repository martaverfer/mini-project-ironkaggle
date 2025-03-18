
# Standard Libraries
import os

# Data Handling & Computation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error 

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

def corr_heatmap(dataframe):

    df = dataframe.copy()

    # Remove date column for the correlation matrix
    if "date" in dataframe.columns:
        df = df.drop(columns=["date"])

    # Moving price columns as last
    df = df[[col for col in df.columns if col != 'price'] + ['price']]

    # Correlation matrix calculation and heatmap
    correlation_matrix = df.corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(13, 10))
    sns.heatmap(correlation_matrix, 
                    cmap="magma_r", 
                    linewidths=0.5, 
                    annot=True, 
                    fmt=".2f",
                    xticklabels=[col.replace('_', ' ').title() for col in df.columns],
                    yticklabels=[col.replace('_', ' ').title() for col in df.columns], 
                    mask=mask
                    )
    plt.title("Correlation Heatmap", fontsize=14)

    # Looking for correlations with the target for printing
    correlation_with_price = df.corrwith(df["price"]).sort_values(ascending=False)
    
    return correlation_with_price

def selecting_features(dataframe, corr_coef=0.25):

    # Remove date column for the correlation matrix
    if "date" in dataframe.columns:
        dataframe = dataframe.drop(columns=["date"])

    # Correlation matrix with price
    correlation_with_price = dataframe.corrwith(dataframe["price"]).sort_values(ascending=False)

    # Saving columns names with a correlation > 0.25
    correlated_columns = correlation_with_price[correlation_with_price >= corr_coef].index.tolist()
    correlated_columns = [col for col in correlated_columns if (col != 'price')]
    
    print(f"Features with correlation coefficient with price > than {round(corr_coef, 2)}")
    if correlated_columns != []:     
        print("Columns that will be used for the training:\n", correlated_columns, "\n") 

    return correlated_columns   

def select_training_set(dataframe, correlated_columns, test_size=0.2):

    # Creating the training dataset
    X = dataframe[correlated_columns]
    y = dataframe['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    #print(f'100% of our data: {len(dataframe)}.')
    #print(f'{int((1-test_size)*100)}% for training data: {len(X_train)}.')
    #print(f'{int(test_size*100)}% for test data: {len(X_test)}.')
    print(f"🔹 Test size {int(test_size * 100)}%:")
    print(f"  Training set size: {len(X_train)} | Test set size: {len(X_test)}")
    

    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, X_test, y_train, y_test, results_list=[]):

    # Initializing and training the Decision Tree Regressor
    model_tree = DecisionTreeRegressor(random_state=42)
    model_tree.fit(X_train, y_train)
    
    # Making predictions
    y_pred = model_tree.predict(X_test)
    
    # Calculating metrics for the model
    MSE = mean_squared_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(y_test, y_pred)
    
    # Print the metrics
    print(f"  Model Metrics: | R2 = {round(R2, 4)} | RMSE = {round(RMSE, 4)} | MAE = {round(MAE, 4)} | MSE = {round(MSE, 4)} |")

    # Create a table with actual vs predicted values and their difference
    results_df = pd.DataFrame({
        "Actual Price": y_test,
        "Predicted Price": y_pred,
        "Difference": y_test - y_pred
    })

    # Provide insights about model performance
    if R2 >= 0.75:
        print("✅ The model performs well! It explains a large portion of the variance.\n")
    elif 0.5 <= R2 < 0.75:
        print("⚠️ The model is moderately good, but there’s room for improvement.\n")
    else:
        print("❌ The model performs poorly. Consider tuning hyperparameters or using a different approach.\n")
    
    # Append all data to results_list as a dictionary
    results_list.append({
            "MSE": MSE,
            "RMSE": RMSE,
            "MAE": MAE,
            "R2": R2
        })    
    
    return results_df, results_list

def create_train_test_splits_and_evaluate(dataframe, correlated_columns):
   
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    results_list = []
    
    for test_size in test_sizes:
        #print(f"\n🔹 Test size {int(test_size * 100)}%:")
        X_train, X_test, y_train, y_test = select_training_set(dataframe, correlated_columns, test_size)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        #print(f"\n🔹 Test size {int(test_size * 100)}%:")
        #print(f"  Training set size: {len(X_train)} | Test set size: {len(X_test)}")
        
        results_table, results_list = train_decision_tree(X_train, X_test, y_train, y_test, results_list)
        results_list[-1]["test_size"] = test_size

    return results_list

def evaluate_different_correlations(dataframe):

    correlation_thresholds = np.arange(0.2, 1.0, 0.05)  
    results = []

    for corr_coef in correlation_thresholds:

        # Select features based on correlation coefficient
        correlated_columns = selecting_features(dataframe, corr_coef)

        # Ensure there are enough features to proceed
        if not correlated_columns:
            print("⚠️ No features meet this correlation threshold. Ending...\n")
            break

        # Train and evaluate models for different test sizes
        results_list = create_train_test_splits_and_evaluate(dataframe, correlated_columns)

        # Collecting results for the DataFrame
        for result in results_list:
            results.append({
                    "Correlation Coefficient ≥": corr_coef,
                    "Test Size (%)": result["test_size"],
                    "R²": result["R2"],
                    "RMSE": result["RMSE"],
                    "MAE": result["MAE"],
                    "MSE": result["MSE"],
            })

    # Convert collected results into a DataFrame
    results_df = pd.DataFrame(results)

    return results_df     

def cross_validate_model(dataframe, correlated_columns, n_splits=5):

    #correlated_columns = selecting_features(dataframe, corr_coef=0.25)

    # Preparing the data
    X = dataframe[correlated_columns]
    y = dataframe['price']
    
    # Setting up KFold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []
    training_r2_scores = []
    test_r2_scores = []
    average = {}
    
    fold = 1
    for train_index, test_index in kf.split(X):

        # Splitting the data into training and testing sets for the fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Initialize and train the model
        model_tree = DecisionTreeRegressor(random_state=42)
        model_tree.fit(X_train, y_train)
        
        # Predict on the training and testing sets
        y_train_pred = model_tree.predict(X_train)
        y_test_pred = model_tree.predict(X_test)
        
        # Calculate R² scores for training and testing
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Append results for each fold
        training_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)
        
        # Save fold results to print and return later
        results.append({
            'Fold': fold,
            'Train R²': round(train_r2, 4),
            'Test R²': round(test_r2, 4)
        })
        
        fold += 1

    # Create a DataFrame to return the results of the cross-validation
    results_df = pd.DataFrame(results)

    # Calculate average R² scores across all folds
    avg_train_r2 = round(sum(training_r2_scores) / len(training_r2_scores), 4)
    avg_test_r2 = round(sum(test_r2_scores) / len(test_r2_scores), 4)

    # Append the results for this n_splits to the overall results
    average = {
            'n_splits': n_splits,
            'Average Train R²': avg_train_r2,
            'Average Test R²': avg_test_r2
        }

    # Print conclusion based on the R² scores
    print("Cross-Validation Results:")
    print(f"Number of folds: {n_splits}")
    print("Average Training R²: ", avg_train_r2)
    print("Average Test R²: ", avg_test_r2)
    
    # Conclusion based on the average test R²
    if avg_test_r2 >= 0.75:
        print("✅ The model performs well on unseen data, explaining a large portion of the variance.\n")
    elif 0.5 <= avg_test_r2 < 0.75:
        print("⚠️ The model is moderately good, but there's room for improvement in its generalization.\n")
    else:
        print("❌ The model performs poorly on unseen data. Consider tuning hyperparameters or using a different approach.\n")
    
    return results_df, average

def tunning_cross_validate_model(dataframe, correlated_columns):

    results_by_n_splits = []
    n_splits_list = [5, 7, 10, 13, 15]

    # Loop through different n_splits values
    for n_splits in n_splits_list:
        results_df, average = cross_validate_model(dataframe, correlated_columns, n_splits)
        results_by_n_splits. append(average)
           
    # Create a DataFrame to return the results of the cross-validation
    results_by_n_splits = pd.DataFrame(results_by_n_splits)
    
    return results_by_n_splits      