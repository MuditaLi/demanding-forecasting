"""
Script to get split of weekly forecast per month
Example: split = split_monthly_forecast_to_weekly(
    [12, 18], pd.to_datetime(pd.Timestamp.today().date()) - np.timedelta64(1, 'Y'))
"""
import pandas as pd
import preprocessor.names as n
import util.misc as misc
from configs.configs import Configs
from preprocessor.data_loader import DataLoader
from dcd_forecast_model.model import ModelFactory
from dcd_forecast_model.trainer import Predictor
from util.splits import get_columns_scenarios


def predict_multiple_horizons(df_predictions_list, configs: Configs, data_loader: DataLoader,
                              scenario_weather=None, scenario_promo: bool=False):
    """
    Args:
        df_predictions_list (pd.DataFrame): columns = 'lead_sku', 'customer_planning_group', 'plant',
        'date_when_predicting', 'date_to_predict'
        min_horizon (int): in weeks. For example, we predict the first week of the month 10 weeks ahead
        configs: single model configs file
        data_loader: all data objects
        scenario_promo (bool): True = include promotion uplift calculation
        scenario_weather (dict): parameters for the weather scenarios to run, e.g.
        {'use_existing_year_data': [2018]} or {'degrees_celsius': [1, 2, 3]} or {} if you don't want to run a scenario

    Returns:

    """
    results = list()
    relevant_columns_merge = [
        n.FIELD_LEAD_SKU_ID,
        n.FIELD_CUSTOMER_GROUP,
        n.FIELD_PLANT_ID,
        'date_when_predicting',
        'date_to_predict'
    ]

    tmp = df_predictions_list[[n.FIELD_PREDICTION_DATE, 'date_when_predicting']].drop_duplicates()
    tmp['forecasting_horizon'] = tmp.apply(lambda x: misc.delta_between_periods(*x), axis=1)
    df_predictions_list = df_predictions_list.merge(
        tmp, on=[n.FIELD_PREDICTION_DATE, 'date_when_predicting'], how='left')

    for horizon in tmp['forecasting_horizon'].unique():
        df_predictions_horizon = (df_predictions_list[df_predictions_list['forecasting_horizon'] == horizon])

        mod = ModelFactory()
        mod.load_model(
            forecasting_horizon=horizon,
            is_sell_in_model=configs.is_sell_in_model,
            country=configs.country
        )
        predictor = Predictor(
            start_test=df_predictions_horizon['date_when_predicting'].min(),
            end_test=df_predictions_horizon['date_when_predicting'].max(),
            model=mod,
            granularity=configs.granularity,
            forecasting_horizons=[horizon],
            feature_controller=configs.features
        )
        results_horizon = predictor.predict(data_loader)

        if scenario_promo:
            results_horizon = predictor.predict_base_line(results_horizon)

        # Run weather scenarios if dictionary is not empty (keep temperature w0 & prediction)
        if scenario_weather:
            results_horizon = predictor.run_weather_scenarios(results_horizon, data_loader, **scenario_weather)

        output_columns = get_columns_scenarios(results_horizon)
        df_predictions_horizon = (df_predictions_horizon
                                  .merge(results_horizon[relevant_columns_merge + list(set(output_columns))],
                                         on=relevant_columns_merge, how='left')
                                  )
        # Remove unknown (cpg, lead_sku, plant) combinations that ML model is not able to predict
        df_predictions_horizon = df_predictions_horizon[df_predictions_horizon[n.FIELD_PREDICTION].notnull()]

        results.append(df_predictions_horizon)

    return pd.concat(results, axis=0, sort=False).drop(columns=['forecasting_horizon'])
