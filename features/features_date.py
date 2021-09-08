import preprocessor.names as n
import pandas as pd
import numpy as np


def create_dates(dates_to_predict):
    """

    """

    df = pd.DataFrame()
    df['date_to_predict'] = dates_to_predict

    df['wk'] = df.date_to_predict.apply(lambda x: (x % 100) / 53)

    df['month_ind'] = df['wk'] * 12
    df['sin_month'] = np.sin(df['wk'] * np.pi)

    return df[['date_to_predict', 'month_ind', 'sin_month']]


def create_seasonality_features(dates_to_predict):

    df = pd.DataFrame()
    df['date_to_predict'] = dates_to_predict

    df['month'] = df.date_to_predict.apply(lambda x: (x % 100) / 12)
    df['sin_month'] = np.sin(df['month'] * np.pi)
    df['cos_month'] = np.cos(df['month'] * np.pi)

    return df[['date_to_predict', 'month', 'sin_month', 'cos_month']]
