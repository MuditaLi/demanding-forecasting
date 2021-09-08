"""
Script to assess month minus 3 forecast accuracy performance of model
Please specify model & evaluation period in configs yaml file
$ python evaluation_m3.py 'default_france.yaml'
"""
import sys
import pandas as pd
from functools import partial
import preprocessor.names as n
import util.misc as misc
from configs.configs import Configs
from .evaluation import Evaluator
from dcd_forecast_model.model import ModelFactory
from preprocessor.data_loader import DataLoader
from dcd_forecast_model.trainer import Predictor


def replace_week_granularity_by_month(granularities: list):

    def helper(l):
        return list(map(lambda x: {n.FIELD_YEARWEEK: n.FIELD_YEARMONTH}.get(x, x), l))

    return list(map(helper, granularities))


def build_m3_target_predictions_table(apo_week_month_list, evaluation_start, evaluation_end, min_horizon: int=10):
    # M-3 EVALUATION --> FORECAST WEEKS 1|2|3|4 of a given month 10|11|12|13 weeks ahead
    in_period = ((apo_week_month_list[n.FIELD_YEARWEEK] >= evaluation_start) &
                 (apo_week_month_list[n.FIELD_YEARWEEK] <= evaluation_end))
    apo_week_month_list = apo_week_month_list[in_period]

    apo_week_month_list['forecasting_horizon'] = apo_week_month_list.groupby(
        n.FIELD_YEARMONTH)[n.FIELD_YEARWEEK].rank().astype(int)

    apo_week_month_list['forecasting_horizon'] += min_horizon - 1
    apo_week_month_list.columns = [n.FIELD_YEARMONTH, 'date_to_predict', 'forecasting_horizon']
    apo_week_month_list['date_when_predicting'] = apo_week_month_list[
        ['date_to_predict', 'forecasting_horizon']
    ].apply(lambda x: partial(misc.substract_period, highest_period=52)(*x), axis=1)

    return apo_week_month_list[[n.FIELD_YEARMONTH, 'date_when_predicting', 'date_to_predict']].reset_index(drop=True)


def predict_multiple_horizons(df_predictions_list, configs: Configs, data_loader: DataLoader, min_horizon: int=10):
    """
    Args:
        df_predictions_list (pd.DataFrame): columns = 'lead_sku', 'customer_planning_group', 'plant',
        'date_when_predicting', 'date_to_predict'
        min_horizon (int): in weeks. For example, we predict the first week of the month 10 weeks ahead
        configs:
        data_loader:

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

    tmp = df_predictions_list[['date_to_predict', 'date_when_predicting']].drop_duplicates()
    tmp['forecasting_horizon'] = tmp.apply(lambda x: misc.delta_between_periods(*x), axis=1)
    df_predictions_list = df_predictions_list.merge(tmp, on=['date_to_predict', 'date_when_predicting'], how='left')

    for horizon in tmp['forecasting_horizon'].unique():

        # We only evaluate the performance on the first 4 weeks of a month
        if horizon > min_horizon + 3:
            continue

        df_predictions_horizon = (df_predictions_list[df_predictions_list['forecasting_horizon'] == horizon])

        mod = ModelFactory()
        mod.load_model(forecasting_horizon=horizon, is_sell_in_model=configs.is_sell_in_model)
        predictor = Predictor(
            start_test=df_predictions_horizon['date_when_predicting'].min(),
            end_test=df_predictions_horizon['date_when_predicting'].max(),
            model=mod,
            granularity=configs.granularity,
            forecasting_horizons=[horizon],
        )
        results_horizon = predictor.predict(data_loader)

        df_predictions_horizon = (df_predictions_horizon
                                  .merge(results_horizon[relevant_columns_merge + [n.FIELD_PREDICTION]],
                                         on=relevant_columns_merge, how='left')
                                  )
        # TODO - Unknown (cpg, lead_sku, plant) combinations, so model is not able to predict right now
        df_predictions_horizon = df_predictions_horizon[df_predictions_horizon[n.FIELD_PREDICTION].notnull()]

        results.append(df_predictions_horizon)

    return pd.concat(results, axis=0, sort=False).drop(columns=['forecasting_horizon'])


def compare_with_historical_m3_forecast(configs: Configs, data_loader: DataLoader):
    """ Function to aggregate our prediction per month & compare with historical forecasts """

    # Historical forecasts
    granularity_week = [
        n.FIELD_YEARWEEK,
        n.FIELD_LEAD_SKU_ID,
        n.FIELD_CUSTOMER_GROUP,
        n.FIELD_PLANT_ID
    ]
    granularity_month = [n.FIELD_YEARMONTH] + granularity_week[1:]

    relevant_columns = [
        n.FIELD_YEARMONTH,
        n.FIELD_YEARWEEK,
        n.FIELD_LEAD_SKU_ID,
        n.FIELD_CUSTOMER_GROUP,
        n.FIELD_PLANT_ID,
        Evaluator.GROUND_TRUTH_COLUMN_APO_FINAL_FORECAST_M3,
        'forecast_volume_m-3'
    ]
    data_loader.load_sell_in_forecast_accuracy_comparison_data(Evaluator.GROUND_TRUTH_APO_FORECAST_ACCURACY_REPORT_M3)

    historical_forecasts = data_loader.df_fc_acc.filter(relevant_columns)
    in_period = ((historical_forecasts[n.FIELD_YEARWEEK] >= configs.evaluation_start) &
                 (historical_forecasts[n.FIELD_YEARWEEK] <= configs.evaluation_end))
    historical_forecasts = historical_forecasts[in_period]

    # Group to get one value per SKU, CPG, Plant, Week
    historical_forecasts = (historical_forecasts
                            .groupby(granularity_week)
                            .agg({n.FIELD_YEARMONTH: 'first',
                                  Evaluator.GROUND_TRUTH_COLUMN_APO_FINAL_FORECAST_M3: 'sum',
                                  'forecast_volume_m-3': 'sum'})
                            .reset_index()
                            )

    # Predictions
    apo_week_month_list = (historical_forecasts[[n.FIELD_YEARMONTH, n.FIELD_YEARWEEK]]
                           .sort_values([n.FIELD_YEARMONTH, n.FIELD_YEARWEEK])
                           .drop_duplicates(n.FIELD_YEARWEEK))
    apo_week_month_list = build_m3_target_predictions_table(
        apo_week_month_list, configs.evaluation_start, configs.evaluation_end)

    # list contains all (SKU, CPG, Plant) combinations that appear at least once in a given month
    # TODO - cross join to get all existing (material, plant, cpg) combinations
    df_predictions_list = historical_forecasts[granularity_month].drop_duplicates()
    df_predictions_list = df_predictions_list.merge(apo_week_month_list, on=n.FIELD_YEARMONTH, how='left')
    predictions = predict_multiple_horizons(df_predictions_list, configs, data_loader)

    # scope historical forecasts table to keep exactly the same weeks as the predicted ones (before aggregating)
    historical_forecasts = historical_forecasts.merge(
        predictions.rename(columns={'date_to_predict': n.FIELD_YEARWEEK}).filter(granularity_week).drop_duplicates(),
        on=granularity_week, how='inner'
    )

    # Comparison ML forecast vs historical forecast
    # predictions_output = pd.read_csv('predictions_m3.csv')
    aggs = {Evaluator.GROUND_TRUTH_COLUMN_APO_FINAL_FORECAST_M3: 'sum', 'forecast_volume_m-3': 'sum'}
    historical_forecasts_month = historical_forecasts.groupby(granularity_month).agg(aggs).reset_index()
    predictions_month = predictions.groupby(granularity_month)[n.FIELD_PREDICTION].sum().reset_index()
    forecasts_apo_vs_ml_m3 = historical_forecasts_month.merge(
        predictions_month,
        on=granularity_month, how='left'
    )

    # Add is_on trade information
    tmp = data_loader.df_cpg_trade_channel[[n.FIELD_CUSTOMER_GROUP, 'trade_type']]
    tmp['is_on_trade'] = (tmp['trade_type'] == 'is_on_trade')
    tmp = tmp[[n.FIELD_CUSTOMER_GROUP, 'is_on_trade']]
    forecasts_apo_vs_ml_m3 = forecasts_apo_vs_ml_m3.merge(tmp, on=n.FIELD_CUSTOMER_GROUP, how='left')

    forecasts_apo_vs_ml_m3 = forecasts_apo_vs_ml_m3[forecasts_apo_vs_ml_m3[n.FIELD_PLANT_ID] != 'F015']
    evaluator = Evaluator(
        data=data_loader,
        granularity=granularity_month,
        start_periods_eval=configs.evaluation_start,
        end_periods_eval=configs.evaluation_end,
        forecasting_horizon=3,
        granularity_eval='forecast_volume_m-3',
        is_sell_in_model=configs.is_sell_in_model,
        is_weekly_forecast=False
    )
    granularities = replace_week_granularity_by_month(configs.granularity_evaluator.values())
    evaluator.report(forecasts_apo_vs_ml_m3, granularities)
    evaluator.detailed_report(forecasts_apo_vs_ml_m3, granularities)


if __name__ == '__main__':

    configs = Configs(sys.argv[1])
    data_loader = DataLoader(configs.is_sell_in_model)
    data_loader.load_data(configs.main_file_name)
    compare_with_historical_m3_forecast(configs, data_loader)
