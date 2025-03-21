
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

import logging

dict_test_results_template = {
    'test_size': '',
    'random_state': '',
    'R2': '',
    'MAE': '',
    'RMSE': '',
    'MSE': '',
}

def linear_regression_test(X, y, test_size, random_state):
    # Test/Train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logging.debug(f'70% for training data: {len(X_train)}.')
    logging.debug(f'30% for test data: {len(X_test)}.')

    # Train the model
    model = LinearRegression()
    model.fit(X_train,y_train)

    # Make predictions on the test dataset
    predictions = model.predict(X_test)
    r2_3 = r2_score(y_test, predictions)
    MAE_3 = mean_absolute_error(y_test, predictions)
    RMSE_3 = root_mean_squared_error(y_test, predictions)
    MSE_3 = mean_squared_error(y_test, predictions)

    #Printing the results
    logging.debug(f"R2 = {round(r2_3, 4)}")
    logging.debug(f"MAE = {round(MAE_3, 4)}")
    logging.debug(f"RMSE = {round(RMSE_3, 4)}")
    logging.debug(f"MSE =  {round(MSE_3, 4)}")

    # collecting test results to compare
    dict_test_results = dict_test_results_template.copy()
    dict_test_results['test_size'] = test_size
    dict_test_results['random_state'] = random_state
    dict_test_results['R2'] = r2_3
    dict_test_results['MAE'] = MAE_3
    dict_test_results['RMSE'] = RMSE_3
    dict_test_results['MSE'] = MSE_3

    return dict_test_results