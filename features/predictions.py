import pandas as pd


def filter_predictions_list(df_predictions_list: pd.DataFrame, dates_to_predict: list):
    idx = ((df_predictions_list['date_to_predict'] >= min(dates_to_predict))
           &
           (df_predictions_list['date_to_predict'] <= max(dates_to_predict))
           )

    return df_predictions_list[idx]
