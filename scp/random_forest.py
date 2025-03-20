# Data Handling & Computation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error 
from sklearn.feature_selection import RFE
from sklearn.model_selection import learning_curve
import scipy.stats as stats
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestRegressor

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

import scp.decision_tree as dt

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
# Visualization Functions
# ==============================

def train_random_forest(X_train, X_test, y_train, y_test, results_list=[], model_tree = "", estimator = 100):
    # Initializing and training the Decision Tree Regressor
    if model_tree == "":
        model_tree = RandomForestRegressor(n_estimators=estimator, random_state=42)
    
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
    
    return results_df, results_list, model_tree

def create_train_test_splits_and_evaluate(dataframe, correlated_columns, estimator = 100):
   
    test_sizes = [0.1, 0.2, 0.3, 0.4]
    results_list = []
    
    for test_size in test_sizes:
        #print(f"\n🔹 Test size {int(test_size * 100)}%:")
        X_train, X_test, y_train, y_test = dt.select_training_set(dataframe, correlated_columns, test_size)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        #print(f"\n🔹 Test size {int(test_size * 100)}%:")
        #print(f"  Training set size: {len(X_train)} | Test set size: {len(X_test)}")
        
        results_table, results_list, model_tree = train_random_forest(X_train, X_test, y_train, y_test, results_list, estimator=estimator)
        results_list[-1]["test_size"] = test_size

    return results_list

def evaluate_different_correlations(dataframe):

    correlation_thresholds = np.arange(0.2, 1.0, 0.05)  
    results = []

    for corr_coef in correlation_thresholds:

        # Select features based on correlation coefficient
        correlated_columns = dt.selecting_features(dataframe, corr_coef)

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

def evaluate_different_estimators(dataframe):

    correlation_thresholds = np.arange(0.2, 0.35, 0.05)  
    results = []
    estimators = [10, 50, 100, 200, 300, 500, 1000]
    for estimator in estimators:
        print(f"Number of tree: {estimator}")
        for corr_coef in correlation_thresholds:

            # Select features based on correlation coefficient
            correlated_columns = dt.selecting_features(dataframe, corr_coef)

            # Ensure there are enough features to proceed
            if not correlated_columns:
                print("⚠️ No features meet this correlation threshold. Ending...\n")
                break

            # Train and evaluate models for different test sizes
            results_list = create_train_test_splits_and_evaluate(dataframe, correlated_columns, estimator=estimator)

            # Collecting results for the DataFrame
            for result in results_list:
                results.append({
                        "Estimator": estimator,
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

def cross_validate_model(dataframe, correlated_columns, n_splits=5, model_tree = ""):

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
        if model_tree == "":
            model_tree = RandomForestRegressor(n_estimators=50, random_state=42)
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

def perform_grid_search(dataframe):

    results_list = []

    n_splits_list = [5]
    test_sizes = [0.2, 0.3, 0.4]
    #correlation_thresholds = np.arange(0.2, 1.0, 0.05)
    correlation_thresholds = [0.25]

    param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4]     # Minimum samples per leaf
    }

    model_tree = RandomForestRegressor(random_state=42)
     
    for corr_limit in correlation_thresholds:
        print("\n" + "="*50) 
        correlated_columns = dt.selecting_features(dataframe, corr_coef=corr_limit)
        print("="*50)

        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = dt.select_training_set(dataframe, correlated_columns, test_size)
       
            for cv in n_splits_list:
                
                print(f"\n  Number of folds {cv}")
                
                # Initialize GridSearchCV
                grid_search = GridSearchCV(model_tree, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    
                # Fit the model
                grid_search.fit(X_train, y_train)
    
                # Get results
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                results_list.append({
                    'Correlation Coef Limit': corr_limit,
                    'Test Size': test_size,
                    'Cross-validation fold numbers': cv,
                    'Best Parameters': best_params,
                    'Best R² Score': round(best_score, 4)
                })

                print(f"\n✅ Best R² Score: {best_score:.4f}")
                print(f"🏆 Best Parameters: {best_params}\n")
                #print("📊 Cross-Validation Results:")
                #print(pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score"))

                if best_score > 0.75:
                    print("\n✅ Excellent model! The fit is strong.")
                elif 0.50 <= best_score <= 0.75:
                    print("\n⚠️ Acceptable model. Consider tuning further.")
                else:
                    print("\n❌ Poor model! Needs improvement.")

    results_df = pd.DataFrame(results_list)
    
    return results_df