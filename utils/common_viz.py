# 📚 Basic libraries
import scipy.stats as stats
import numpy as np 

# 📊 Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# 🤖 Machine Learning
from sklearn.model_selection import learning_curve, validation_curve


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

