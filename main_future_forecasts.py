"""
Script to generate all future forecasts for a given date_when_predicting
Configs must be specify in configs/default_future_forecasts_france.py

To run the script:
python main_future_forecasts.py default_future_forecasts_france.yaml

Data needed:
To run that script, provide all relevant data, including sales for last 52 weeks before first_date_to_predict
and open orders data until last_date_to_predict (or at least a list of combinations to predict in
DataLoader.df_predictions_list).
"""
import os
import sys
import time
import pandas as pd
from preprocessor.data_loader import DataLoader
from configs.configs import ConfigsFuturePredictor, Configs
from dcd_forecast_model.multiple_predictor import predict_multiple_horizons
from util.splits import split_monthly_forecast_to_weekly, get_columns_predictions
from util.logger import Logger
from util.paths import PATH_OUTPUTS
from util.input_output import write_csv


def save_predictions_to_file(configs: ConfigsFuturePredictor, df_output: pd.DataFrame, country_name: str=''):
    """ Save output in outputs folder """
    model_type = 'in' if configs.is_sell_in_model else 'out'
    now = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M')
    scope = '_'.join([str(configs.first_date_to_predict), str(configs.last_date_to_predict)])
    file_name = '_'.join([now, 'forecasts', 'sell_' + model_type, scope + '.csv'])

    folder = str(configs.date_when_predicting)
    path = os.path.join(PATH_OUTPUTS, country_name, folder[:4], folder[4:], file_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_csv(df_output, path)
    Logger.info('Outputs saved at', path, __file__)


def check_for_discrepancies_in_output(predictions_list, predictions_ml):
    """ Test function to check data outputted by the main_future_forecast function"""

    def helper(suffix_message: str, combinations: pd.MultiIndex):
        """ helper function to format a pandas multi_index object to a string table and output it in Logger """
        df = combinations.to_frame().reset_index(drop=True)
        if not df.empty:
            Logger.warning('The following combinations ' + suffix_message, '\n' + df.to_string())

    combinations_to_predict = predictions_list.set_index(list(predictions_list.columns)).index
    combinations_predicted = predictions_ml.set_index(list(predictions_list.columns)).index

    missing_combinations = combinations_to_predict.difference(combinations_predicted)
    helper('should have been predicted, but are missing', missing_combinations)

    # should always be empty, but let's double check anyway
    unwanted_combinations = combinations_predicted.difference(combinations_to_predict)
    helper('were predicted, but are surplus to requirement', unwanted_combinations)


def main_future_forecast(configs: ConfigsFuturePredictor):
    """
    Main function to run the future forecast for a whole horizon

    1. Build the prediction list containing all date_when_predicting, date_to_predict, lead_sku,
    customer_planning_group, plant combinations to predict

    2. Use multiple_predictor to load and predict relevant volumes for all horizons of interest

    (Optional: 3. Split weekly forecast by month with split_monthly_forecast_to_weekly)

    Args:
        configs (ConfigsFuturePredictor): Configs element needed to run a future forecast

    Returns:
        - predictions (including sceanrios specified)
        - name of country

    """
    # TODO - OPEN ORDERS FILE NAME should be flexible too

    configs_model = Configs(
        config_file_name=configs.file_name_configs_model,
        is_sell_in_model=configs.is_sell_in_model
    )
    data_loader = DataLoader(configs_model, configs_model.features)
    data_loader.load_data(configs_model.main_file_name)

    df_predictions_list = data_loader.df_predictions_list
    if df_predictions_list is None or df_predictions_list.empty:
        df_predictions_list = data_loader.generate_default_predictions_list()

    df_predictions_list = df_predictions_list[
        (df_predictions_list['date_to_predict'] >= configs.first_date_to_predict)
        &
        (df_predictions_list['date_to_predict'] <= configs.last_date_to_predict)
    ]
    df_predictions_list['date_when_predicting'] = configs.date_when_predicting

    # Use ML models to predict sales volume
    predictions = predict_multiple_horizons(df_predictions_list, configs_model, data_loader,
                                            configs.scenario_weather, configs.scenario_promo)

    # Data check to ensure that we did predict all combinations provided in the predictions list
    check_for_discrepancies_in_output(df_predictions_list, predictions)

    return predictions, configs_model.country


if __name__ == '__main__':

    # sys.argv[1] = 'default_future_forecasts_france.yaml'

    start = time.time()
    Logger.info('Future Forecast Predictions', 'start', __file__)
    configs_forecast = ConfigsFuturePredictor(sys.argv[1])

    # Step 1: Run all future forecasts (output of model only)
    df_future_forecasts, country = main_future_forecast(configs_forecast)

    # Step 2: Apply monthly split to get a forecast per week
    columns_predictions = get_columns_predictions(df_future_forecasts)
    df_future_forecasts_split = split_monthly_forecast_to_weekly(
        configs_forecast.date_when_predicting, df_future_forecasts, col_predictions=columns_predictions)
    Logger.info('End Future Forecast Predictions', 'ended in %d seconds' % round((time.time() - start), 1), __file__)

    # Save output to outputs folder
    save_predictions_to_file(configs_forecast, df_future_forecasts_split, country)
