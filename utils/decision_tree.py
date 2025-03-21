
# Standard Libraries
import os

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
FONT_PATH = '../utils/fonts/Montserrat-Regular.ttf'

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
    plt.xticks(rotation=90)
    plt.gca().tick_params(colors='black', labelsize=10, labelcolor='black', which='both', width=2)
   # plt.gca().tick_params(axis='both', which='major', labelsize=10, labelcolor='white', labelrotation=0, length=6, width=2)

    plt.savefig("../images/correlation_heatmap.png", 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True) 

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

def train_decision_tree(X_train, X_test, y_train, y_test, results_list=[], model_tree = ""):

    # Initializing and training the Decision Tree Regressor
    if model_tree == "":
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
    
    return results_df, results_list, model_tree

def create_train_test_splits_and_evaluate(dataframe, correlated_columns):
   
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    results_list = []
    
    for test_size in test_sizes:
        #print(f"\n🔹 Test size {int(test_size * 100)}%:")
        X_train, X_test, y_train, y_test = select_training_set(dataframe, correlated_columns, test_size)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        #print(f"\n🔹 Test size {int(test_size * 100)}%:")
        #print(f"  Training set size: {len(X_train)} | Test set size: {len(X_test)}")
        
        results_table, results_list, model_tree = train_decision_tree(X_train, X_test, y_train, y_test, results_list)
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

def perform_grid_search(dataframe):

    results_list = []

    n_splits_list = [5]
    test_sizes = [0.1, 0.2, 0.3, 0.4]
    #correlation_thresholds = np.arange(0.2, 1.0, 0.05)
    #correlation_thresholds = [0.25]
    correlation_thresholds = [0.2]

    param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
    }

    model_tree = DecisionTreeRegressor(random_state=42)
     
    for corr_limit in correlation_thresholds:
        print("\n" + "="*50) 
        correlated_columns = selecting_features(dataframe, corr_coef=corr_limit)
        print("="*50)

        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = select_training_set(dataframe, correlated_columns, test_size)
       
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
                print(f"🏆 Best Parameters: {best_params}")
                #print("📊 Cross-Validation Results:")
                #print(pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score"))

                if best_score > 0.75:
                    print("✅ Excellent model! The fit is strong.\n")
                elif 0.50 <= best_score <= 0.75:
                    print("⚠️ Acceptable model. Consider tuning further.\n")
                else:
                    print("❌ Poor model! Needs improvement.\n")

    results_df = pd.DataFrame(results_list)
    
    return results_df

def feature_score(final_model, X_train):

    feature_importances = final_model.feature_importances_

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 5))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='#eeba30')
    plt.xlabel('Feature Importance', fontweight='bold', fontsize=12, color='black')
    plt.ylabel('Feature Name', fontweight='bold', fontsize=12, color='black')
    plt.title('Feature Importance', fontweight='bold', fontsize=14, color='black')

    plt.gca().invert_yaxis()
    plt.gca().set_yticklabels([label.replace('_', ' ').title() for label in importance_df['Feature']], color='black')
    plt.gca().tick_params(colors='black', labelsize=10, labelcolor='black', which='both', width=2)
    plt.gca().tick_params(axis='both', which='major', labelsize=10, labelcolor='black', labelrotation=0, length=6, width=2)
    plt.grid(color='black', linestyle='--', linewidth=0.5)

    plt.savefig("../images/feature_importance.png", 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)

    return importance_df

def feature_selection(final_model, X_train, X_test, y_train, y_test, features = 5):

    # Initialize RFE and select the top 5 features
    rfe = RFE(estimator=final_model, n_features_to_select=features)
    X_train_rfe = rfe.fit_transform(X_train, y_train)

    # Fit the model with the selected features
    final_model.fit(X_train_rfe, y_train)

    # Test the model
    X_test_rfe = rfe.transform(X_test)
    y_pred = final_model.predict(X_test_rfe)

    # Evaluate the model
    print(f"Selected Features: {X_train.columns[rfe.support_]}")
    correlated_columns = X_train.columns[rfe.support_]

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

    return correlated_columns, results_df, final_model

def selecting_features_log(dataframe, corr_coef=0.25):

    # Remove date column for the correlation matrix
    if "date" in dataframe.columns:
        dataframe = dataframe.drop(columns=["date"])

    if 'price' in dataframe.columns:
        dataframe = dataframe.drop(columns=['price'])

    # Correlation matrix with price
    correlation_with_price = dataframe.corrwith(dataframe["log_price"]).sort_values(ascending=False)

    # Saving columns names with a correlation > 0.25
    correlated_columns = correlation_with_price[correlation_with_price >= corr_coef].index.tolist()
    correlated_columns = [col for col in correlated_columns if (col != 'log_price')]
    
    print(f"Features with correlation coefficient with price > than {round(corr_coef, 2)}")
    if correlated_columns != []:     
        print("Columns that will be used for the training:\n", correlated_columns, "\n") 

    return correlated_columns   

def select_training_set_log(dataframe, correlated_columns, test_size=0.2):

    # Creating the training dataset
    X = dataframe[correlated_columns]
    y = dataframe['log_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    #print(f'100% of our data: {len(dataframe)}.')
    #print(f'{int((1-test_size)*100)}% for training data: {len(X_train)}.')
    #print(f'{int(test_size*100)}% for test data: {len(X_test)}.')
    print(f"🔹 Test size {int(test_size * 100)}%:")
    print(f"  Training set size: {len(X_train)} | Test set size: {len(X_test)}")
    

    return X_train, X_test, y_train, y_test

def train_decision_tree_log(X_train, X_test, y_train, y_test, results_list=[], model_tree = ""):

    # Initializing and training the Decision Tree Regressor
    if model_tree == "":
        model_tree = DecisionTreeRegressor(random_state=42)
    
    model_tree.fit(X_train, y_train)
    
    # Making predictions
    y_pred = np.exp(model_tree.predict(X_test))
    y_test = np.exp(y_test) 

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

def cross_validate_model_log(dataframe, correlated_columns, n_splits=5, model_tree = ""):

    #correlated_columns = selecting_features(dataframe, corr_coef=0.25)

    # Preparing the data
    X = dataframe[correlated_columns]
    y = dataframe['log_price']
    
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
            model_tree = DecisionTreeRegressor(random_state=42)
        model_tree.fit(X_train, y_train)
        
        # Predict on the training and testing sets
        y_train_pred = model_tree.predict(X_train)
        y_test_pred = model_tree.predict(X_test)
        
        # Calculate R² scores for training and testing
        train_r2 = r2_score(np.exp(y_train), np.exp(y_train_pred))
        test_r2 = r2_score(np.exp(y_test), np.exp(y_test_pred))
        
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

def model_validation(results_df, final_model, X_train, X_test, y_train, y_test):
    """
    The function `model_validation` generates various diagnostic plots and learning/validation curves
    for evaluating a regression model's performance.
    
    Parameters:
        - results_df: The `results_df` parameter is a DataFrame containing the following columns: 
                    ["Actual Price", "Predicted Price", "Difference"]. 
        - final_model: The trained machine learning model that you want to evaluate and validate. 
        - X_train: X_train is the training data features used to fit the model. 
        - X_test: The feature matrix representing the independent variables of the test dataset. 
        - y_train: The target variable (output) from the training dataset.
        - y_test: The actual target values from the test dataset.
    """

    y_actual = results_df['Actual Price']  
    y_pred = results_df['Predicted Price']
    residuals = results_df['Difference']

    # 1. Actual vs. Predicted Values
    plt.figure(figsize=(8, 6))
    sns.regplot(x='Actual Price', y='Predicted Price', data=results_df,
                scatter_kws={"color": "#eeba30", "alpha": 0.5},
                line_kws={"color": "#ae0001", "linewidth": 3},
                ci=100)
    plt.ylim(bottom=0)
    plt.title('Actual vs. Predicted Values', color='black', fontsize=14, fontweight='bold')
    plt.xlabel('Actual', fontweight='bold', fontsize=12, color='black')
    plt.ylabel('Predictions', fontweight='bold', fontsize=12, color='black')

    plt.gca().tick_params(colors='black', labelsize=10, labelcolor='black', which='both', width=2)
    plt.gca().tick_params(axis='both', which='major', labelsize=10, labelcolor='black', labelrotation=0, length=6, width=2)
    plt.grid(color='black', linestyle='--', linewidth=0.5)

    plt.savefig("../images/actualvspredicted.png", 
                bbox_inches='tight', 
                facecolor='none', 
                transparent=True)
    plt.show()

    # 2. Residuals vs. Predicted Values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Predicted Values")
    plt.show()

    # 3. Histogram of Residuals
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=60, kde=True)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    plt.show()

    # 4. Q-Q Plot to check normality of residuals
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.show()

    # Compute the learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        final_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.2, 0.7, 10)
    )

    # Compute the mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # 5. Plot the learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, -train_mean, 'o-', color="blue", label="Training Error")
    plt.plot(train_sizes, -val_mean, 'o-', color="red", label="Validation Error")

    plt.fill_between(train_sizes, -train_mean - train_std, -train_mean + train_std, alpha=0.1, color="blue")
    plt.fill_between(train_sizes, -val_mean - val_std, -val_mean + val_std, alpha=0.1, color="red")

    plt.xlabel("Training Size")
    plt.ylabel("Mean Squared Error")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

    # Define the hyperparameter to tune
    param_range = range(1, 30)

    # Compute the validation curve
    train_scores, val_scores = validation_curve(
        final_model, X_train, y_train, param_name="max_depth",
        param_range=param_range, cv=5, scoring='neg_mean_squared_error'
    )

    # Compute the mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # 6. Plot the validation curve
    plt.figure(figsize=(8, 5))
    plt.plot(param_range, -train_mean, 'o-', color="blue", label="Training Error")
    plt.plot(param_range, -val_mean, 'o-', color="red", label="Validation Error")

    plt.fill_between(param_range, -train_mean - train_std, -train_mean + train_std, alpha=0.1, color="blue")
    plt.fill_between(param_range, -val_mean - val_std, -val_mean + val_std, alpha=0.1, color="red")

    plt.xlabel("Max Depth")
    plt.ylabel("Mean Squared Error")
    plt.title("Validation Curve")
    plt.legend()
    plt.show()

def perform_grid_search_log(dataframe):

    dataframe = dataframe.drop(columns=['price'])
    results_list = []

    n_splits_list = [5]
    test_sizes = [0.1, 0.2, 0.3, 0.4]
    #correlation_thresholds = np.arange(0.2, 1.0, 0.05)
    #correlation_thresholds = [0.25]
    correlation_thresholds = [0.2]

    param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
    }

    model_tree = DecisionTreeRegressor(random_state=42)

    for corr_limit in correlation_thresholds:
        print("\n" + "="*50) 
        correlated_columns = selecting_features_log(dataframe, corr_coef=corr_limit)
        print("="*50)

        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = select_training_set_log(dataframe, correlated_columns, test_size)
       
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
                print(f"🏆 Best Parameters: {best_params}")
                #print("📊 Cross-Validation Results:")
                #print(pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score"))

                if best_score > 0.75:
                    print("✅ Excellent model! The fit is strong.\n")
                elif 0.50 <= best_score <= 0.75:
                    print("⚠️ Acceptable model. Consider tuning further.\n")
                else:
                    print("❌ Poor model! Needs improvement.\n")

    results_df = pd.DataFrame(results_list)
    
    return results_df

def feature_selection_log(final_model, X_train, X_test, y_train, y_test, features = 5):

    # Initialize RFE and select the top 5 features
    rfe = RFE(estimator=final_model, n_features_to_select=features)
    X_train_rfe = rfe.fit_transform(X_train, y_train)

    # Fit the model with the selected features
    final_model.fit(X_train_rfe, y_train)

    # Test the model
    X_test_rfe = rfe.transform(X_test)
    y_pred = np.exp(final_model.predict(X_test_rfe))

    # Evaluate the model
    print(f"Selected Features: {X_train.columns[rfe.support_]}")
    correlated_columns = X_train.columns[rfe.support_]

    y_test = np.exp(y_test)
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

    return correlated_columns, results_df, final_model
