## Template
- [ ] Task not done
- [x] Task completed

## Feedback from Isi:
- [ ] Add formated output for error metrics with '$' sign
- [ ] Feature importance remove from plot name and keep only the '... coefficients'
- [ ] Isi accepts feature importance only for one model, which is... "???"

## Generic tasks

These tasks have the scope of the entire project.

- [ ] Make filepaths adjustable for Win and Mac users.
- [ ] Reduse excessive library imports.
- [ ] Fix overwritting exiting files.
- [ ] Logger: either use it everywhere or not use it at all.

# For notebooks

## Feature engineering
- [ ] See the shap feature importance in `feature_engineering.ipynb` to compare with `random_forest_reg.ipynb`
- [ ] Remove feature importance from this notebook, maybe?
- [ ] Unify features for all models to get more precise comparison of the models.

## Random forest regressor
- [ ] Optimize performance.
- [ ] Reduce code duplication.

## Desicison tree
- [ ] Reduce code duplication.
- [ ] Add markdown headers to highlight iterations of model improvements.

## Ridge regression
- [ ] Optimize performance.

## For linear regression

- [x] Move model test to a python script

- [ ] Add more parameters for  hyperparameter test
  - [ ] fit_intercept: bool = True,
  - [ ] copy_X: bool = True,
  - [ ] n_jobs: Int | None = None,
  - [ ] positive: bool = False

- [ ] Finalize visualisation for features by grouping them by topic and add barplots or donut plots with frequency tables for categorical data.

- [ ] Visualize results using Annie's method:

```python 
def model_validation(results_df, final_model):
    """
    The function `model_validation` generates various diagnostic plots and learning/validation curves
    for evaluating a regression model's performance.
    
    Parameters:
        - results_df: The `results_df` parameter is a DataFrame containing the following columns: 
                    ["Actual Price", "Predicted Price", "Difference"]. 
        - final_model: The trained machine learning model that you want to evaluate and validate. 
    """

    y_actual = results_df['Actual Price']  
    y_pred = results_df['Predicted Price']
    residuals = results_df['Difference']

    # 1. Actual vs. Predicted Values
    plt.figure(figsize=(8, 6))
    sns.regplot(x='Actual Price', y='Predicted Price', data=results_df,
                scatter_kws={"color": "blue", "alpha": 0.5},
                line_kws={"color": "red", "linewidth": 3},
                ci=100)
    plt.ylim(bottom=0)
    plt.title('Actual vs. Predicted Values')
    plt.xlabel('Actual')
    plt.ylabel('Predictions')
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

```

# For python modules

## Common functions
- [ ] Save models to pickles

## generic for libs
- [ ] All methids in a long module should be placed alphabetically to find them quickly
- [ ] All functions and clas methods should have doc strings describing parameters and output

## plots
- [ ] Add code to plot class to automatically save all plots as image files:
```python
plt.savefig("your_name.png", dpi=300, transparent=True)
```
- [ ] Add Gryffindor color scale, use approach from `ridge_reg.ipynb`.



# Refactoring notes

## methods to notebooks mapping

module | method | notebook | remarks
--- | --- | --- | ---
decision_tree.py | corr_heatmap | decision_tree.ipynb | TBD
decision_tree.py | create_train_test_splits_and_evaluate | decision_tree.ipynb | TBD
decision_tree.py | cross_validate_model | random_forest_reg.ipynb | TBD
decision_tree.py | cross_validate_model | decision_tree.ipynb | TBD
decision_tree.py | cross_validate_model_log | decision_tree.ipynb | TBD
decision_tree.py | evaluate_different_correlations | decision_tree.ipynb | TBD
decision_tree.py | feature_score | random_forest_reg.ipynb | TBD
decision_tree.py | feature_score | decision_tree.ipynb | TBD
decision_tree.py | feature_selection | random_forest_reg.ipynb | TBD
decision_tree.py | feature_selection | decision_tree.ipynb | TBD
decision_tree.py | feature_selection_log | decision_tree.ipynb | TBD
decision_tree.py | model_validation | decision_tree.ipynb | TBD
decision_tree.py | model_validation | random_forest_reg.ipynb | TBD
decision_tree.py | perform_grid_search | decision_tree.ipynb | TBD
decision_tree.py | perform_grid_search_log | decision_tree.ipynb | TBD
decision_tree.py | select_training_set | decision_tree.ipynb | TBD
decision_tree.py | select_training_set | random_forest_reg.ipynb | TBD
decision_tree.py | select_training_set_log | decision_tree.ipynb | TBD
decision_tree.py | selecting_features | random_forest_reg.ipynb | TBD
decision_tree.py | selecting_features | decision_tree.ipynb | TBD
decision_tree.py | selecting_features_log | decision_tree.ipynb | TBD
decision_tree.py | train_decision_tree | decision_tree.ipynb | TBD
decision_tree.py | train_decision_tree_log | decision_tree.ipynb | TBD
decision_tree.py | tunning_cross_validate_model | decision_tree.ipynb | TBD
lasso_model.py | regression_metrics | xgboost_reg.ipynb | TBD
lasso_model.py | regression_metrics | lasso_reg.ipynb | TBD
lasso_model.py | regression_validation | xgboost_reg.ipynb | TBD
lasso_model.py | regression_validation | lasso_reg.ipynb | TBD
lasso_model.py | reporting_dataframe | xgboost_reg.ipynb | TBD
lasso_model.py | reporting_dataframe | lasso_reg.ipynb | TBD
lasso_model.py | test_train_r_analysis | xgboost_reg.ipynb | TBD
lasso_model.py | test_train_r_analysis | lasso_reg.ipynb | TBD
random_forest.py | create_train_test_splits_and_evaluate | random_forest_reg.ipynb | TBD
random_forest.py | cross_validate_model | random_forest_reg.ipynb | TBD
random_forest.py | evaluate_different_correlations | random_forest_reg.ipynb | TBD
random_forest.py | evaluate_different_estimators | random_forest_reg.ipynb | TBD
random_forest.py | perform_grid_search | random_forest_reg.ipynb | TBD
random_forest.py | train_random_forest | random_forest_reg.ipynb | TBD

  