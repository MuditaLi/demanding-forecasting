"""
Script to generate all forecast to back-test the performance of our model at the monthly level.
That script is only relevant for the sell-in model since we don't have weekly sell-out predictions.

To run the script:
python main_future_forecasts.py 201801 201812 3 default_future_forecasts_france.yaml
i.e. python main_future_forecasts.py first_month_to_predict last_month_to_predict horizon_months config_file_name
"""
import sys
import pandas as pd
from functools import partial
import preprocessor.names as n
from preprocessor.data_loader import DataLoader
from configs.configs import Configs
from dcd_forecast_model.multiple_predictor import predict_multiple_horizons
from util.misc import diff_period, substract_period, delta_between_periods
from util.splits import compute_technical_periods_monthly_split, get_month_nunber


def assign_date_when_predicting(monthly_split: pd.DataFrame, first_month_to_predict: int, last_month_to_predict: int,
                                forecasting_horizon_months: int) -> pd.DataFrame:
    """
    Function to compute the date when predicting for a monthly forecast. Useful for back-testing purposes

    Args:
         monthly_split (pd.DataFrame): List of weeks in each month according to APO split

    Returns:
        monthly_split (pd.DataFrame): Updated list of all (date_to_predict, date_when_predicting) combinations
        scoped, i.e. only contains weeks in the time period we want to predict.
    """
    df_month_weeks = monthly_split[
        (monthly_split[n.FIELD_YEARMONTH] >= first_month_to_predict)
        &
        (monthly_split[n.FIELD_YEARMONTH] <= last_month_to_predict)
    ]

    df_month_weeks['date_when_predicting'] = df_month_weeks[n.FIELD_YEARMONTH].apply(
        lambda x: partial(substract_period, add=forecasting_horizon_months, highest_period=12)(x))

    # Assumption: date_when_predicting = last full week of month - m
    tmp = (monthly_split
           .sort_values([n.FIELD_YEARMONTH, n.FIELD_YEARWEEK], ascending=False)
           .drop_duplicates(n.FIELD_YEARWEEK)
           .groupby(n.FIELD_YEARMONTH)[n.FIELD_YEARWEEK].first().to_dict()
           )
    df_month_weeks['date_when_predicting'] = df_month_weeks['date_when_predicting'].map(tmp)
    # df_month_weeks['forecasting_horizon'] = df_month_weeks[[n.FIELD_YEARWEEK, 'date_when_predicting']].apply(
    #     lambda x: misc.delta_between_periods(*x), axis=1)

    return df_month_weeks.rename({n.FIELD_YEARWEEK: 'date_to_predict'})


def construct_exhaustive_list_of_date_to_predict_date_when_predicting_combinations(
        first_month_to_predict: int, last_month_to_predict: int, horizon_months: int) -> pd.DataFrame:
    """
    Exhaustive list of all date_when_predicting, date_to_predict, ratio to split our predictions
    """
    date_when_predicting = get_month_nunber(
        [pd.to_datetime(str(first_month_to_predict), format='%Y%M') - pd.offsets.MonthBegin(horizon_months)]).astype(int)
    max_horizon = diff_period(date_when_predicting, last_month_to_predict, is_weekly=False)
    monthly_split = compute_technical_periods_monthly_split([0, max_horizon], date_when_predicting)

    relevant_columns = [
        'date_when_predicting',  # in weeks
        'date_to_predict',       # in weeks
        n.FIELD_YEARMONTH,
        'ratio'
    ]
    return (assign_date_when_predicting(monthly_split, first_month_to_predict, last_month_to_predict, horizon_months)
            .filter(relevant_columns))


def monthly_sell_in_forecast(first_month_to_predict: int, last_month_to_predict: int, horizon_months: int,
                             config_file_name: str='default_france.yaml'):
    """
    Main function to assess the performance of our model at a given monthly horizon.

    Args:
        first_month_to_predict (int): format = YYYYMM
        last_month_to_predict (int): format = YYYYMM
        horizon_months (int): single horizon in months
        config_file_name (str): name of single model configs file

    Returns:
        df_predictions (pd.DataFrame): Prediction splitted by month
    """
    configs = Configs(config_file_name=config_file_name, is_sell_in_model=True)
    data_loader = DataLoader(configs, configs.features)
    data_loader.load_data(configs.main_file_name)

    df_predictions_list = data_loader.df_predictions_list
    if df_predictions_list is None or df_predictions_list.empty:
        df_predictions_list = data_loader.generate_default_predictions_list()

    monthly_split = construct_exhaustive_list_of_date_to_predict_date_when_predicting_combinations(
        first_month_to_predict, last_month_to_predict, horizon_months)

    df_predictions_list = df_predictions_list.merge(monthly_split, on=['date_to_predict'], how='right')

    # Predict with multiple horizon
    df_predictions = predict_multiple_horizons(df_predictions_list, configs, data_loader)
    df_predictions[n.FIELD_PREDICTION] = df_predictions[n.FIELD_PREDICTION] * df_predictions['ratio']

    return df_predictions


if __name__ == '__main__':

    # sys.argv[1:] = first_month_to_predict, last_month_to_predict, horizon_months, config_file_name
    monthly_predictions = monthly_sell_in_forecast(*sys.argv[1:])
