# Mini Project: IronKaggle. 

## Project Overview

This repository contains a mini-project which explores how data-driven models and machine learning techniques can predict housing prices in King Country. By analyzing key features we uncover key insights that help improve price estimation accuracy—benefiting buyers, sellers, and investors alike.

## Objectives 

The goal is to develop and evaluate **regression models** to predict house prices in King County by integrating robust data preprocessing, feature engineering, model tuning, and performance comparison to uncover key insights and establish best practices for real-world predictive modeling.

## Methodology

1. **Load and preprocess of the dataset**
    - [Explorotary Data Anbalysis](./docs/eda_report.md) - here you can find the dataset columns overview.
    - Feature Engineering
    - Normalize or standarize numerical values to ensure compatibility for modeling.
2. **Exploratory Data Analysis (EDA):**
    - Visualization techniques to better inderstand the distribution and correlation between the variables.
3. **Regression Models Development:**
    - Linear 
    - Ridge
    - Lasso
    - XGBoost
    - Decision Tree
    - Random Forest
4. **Model Fine-Tuning:**
    - Regularization
    - Hyperparameter tuning
    - Cross validation
5. **Metrics Comparison. Choosing the Best Model:**
    - R-squared
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)

## Analysis and Results
Based on our evaluation of different regression models, the following insights were derived:

- **XGBoost Regression** showed the best overall performance with a high R² test score (0.806) and relatively low RMSE.

- **Random Forest Regression** also performed well, with an R² test score of 0.8004 and good generalization capabilities.

- **Lasso Regression** was effective in feature selection, reducing model complexity without significant loss of accuracy.

- **Decision Trees** tend to overfit unless hyperparameter tuning is applied, as seen in the varying R² scores across cross-validation.

- **Linear and Ridge Regression** models performed moderately well but lacked the predictive power of ensemble methods like XGBoost and Random Forest.

- By analyzing every model, can be concluded that the **features that impact the most in the house price** are the overall grade given to the house, based on the King County grading system, and the square footage of the interior living space. These features consistently appeared as top predictors across multiple models, significantly influencing price estimations.

## Conclusion

In real estate analysis, accurate price predictions are essential for decision-making. The use of advanced models like **XGBoost** and **Random Forest**, along with careful feature selection, hyperparameter tuning, and data preprocessing, enables stakeholders to make better, data-driven decisions. Using these models can significantly enhance pricing strategies, investment decisions, and market insights, ultimately leading to better outcomes in the real estate market.

## Additional content

- [Presentation](./presentations/Presented_by_Gryffindor.pdf)
- [Developer setup](./docs/setup.md)
- [Developer leftover tasks](./docs/TODOs.md)