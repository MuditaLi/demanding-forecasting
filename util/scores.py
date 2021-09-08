"""
Module containing functions to evaluate models
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def smape(predict, actual, weights=None):

    if weights is None:
        weights = np.ones(len(predict))

    num = np.abs(predict - actual) * weights
    den = np.abs(predict) + np.abs(actual)
    den[den == 0] = 1
    evals = num / den

    result = evals.sum()

    if weights.sum() != 0:
        result = result / weights.sum()

    return result


def print_scores(actual, predicted, weights=None):
    """
    """
    mae = mean_absolute_error(actual, predicted, weights)
    mse = mean_squared_error(actual, predicted, weights)
    smape_value = smape(predicted, actual, weights)
    r2 = r2_score(actual, predicted, weights)

    print(f"SMAPE: {smape_value:.2f}")
    print(f"MAE:   {mae:.2f}")
    print(f"RMSE:  {np.sqrt(mse):.2f}")
    print(f"R2:    {r2:.2f}")


def perso_metric(actuals, pred, acceptable_perc=0.2, acceptable_offset=1):
    """
    Personalized metric describing percentage of values in an acceptable 'cone'
    around actual values
    """

    lower_bound = actuals * (1 - acceptable_perc) - acceptable_offset
    upper_bound = actuals * (1 + acceptable_perc) + acceptable_offset

    low = np.mean(pred < lower_bound)
    between = np.mean((pred <= upper_bound) & (pred >= lower_bound))
    high = np.mean(pred > upper_bound)

    return low, between, high


def carlsberg_score(actual_values, predicted_values):
    if not predicted_values.sum():
        return np.nan

    bias = predicted_values - actual_values
    return 1 - (np.abs(bias).sum() / predicted_values.sum())


def bias_score(actual_values, predicted_values):
    if not predicted_values.sum():
        return np.nan

    bias = predicted_values - actual_values
    return bias.sum() / predicted_values.sum()


def root_mean_squared_error(actuals, predicted):
    return np.sqrt(mean_squared_error(actuals, predicted))


def custom_asymmetric_train(y_true, y_pred):
    """
    Asymmetric train loss.
    Factor of difference between under and over forecast currently set to 1.2, but can and should be tested
    in more details.
    """
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual > 0, - 2 * 1.2 * residual, -2 * residual)
    hess = np.where(residual > 0, 2 * 1.2, 2.0)

    return grad, hess
