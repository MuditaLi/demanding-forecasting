"""
The entry point of the demand forecasting model.
"""
import sys
import time
from configs.configs import Configs
from evaluator.evaluation import Evaluator
from dcd_forecast_model.trainer import Predictor, Trainer
import preprocessor.names as n
from preprocessor.data_loader import DataLoader
from dcd_forecast_model.model import ModelFactory
from util.logger import Logger
import util.scores as scores


def main(configs: Configs=None, data_loader: DataLoader=None):
    """
    Main function of the demand forecasting model

    Usage example:
        > python carlsberg/main.py
    """
    # Load data
    if configs is None:
        configs = Configs('default_france.yaml')

    # data_loader = None
    if data_loader is None:
        data_loader = DataLoader(configs, configs.features)
        data_loader.load_data(configs.main_file_name)

    if configs.train_model:
        # Model training
        tr = Trainer(
            start_train=configs.train_start,
            end_train=configs.train_end,
            granularity=configs.granularity,
            feature_controller=configs.features,
            forecasting_horizons=configs.forecasting_horizons,
            is_weekly_forecast=configs.is_weekly_forecast,
            is_sell_in_model=configs.is_sell_in_model,
            use_light_regressor=configs.use_light_regressor,
            # custom_loss_function=scores.custom_asymmetric_train  # Manually switch on custom loss if needed
        )
        mod, train = tr.fit(data_loader)
    else:
        mod = ModelFactory(use_light_regressor=configs.use_light_regressor)
        mod.load_model(country=configs.country,
                       file_name=configs.trained_model_file,
                       forecasting_horizon=configs.forecasting_horizons[0],
                       is_sell_in_model=configs.is_sell_in_model)

    if configs.test_end is not None:
        predictor = Predictor(
            start_test=configs.test_start,
            end_test=configs.test_end,
            model=mod,
            granularity=configs.granularity,
            forecasting_horizons=configs.forecasting_horizons
        )
        res = predictor.predict(data_loader)
        # res = predictor.predict_base_line(res)  # Compute the estimated promo uplift
        # res_weather = predictor.run_weather_scenarios(res, data_loader, degrees_celsius=[1, 2, 0])

        if configs.evaluation_start is not None:
            evaluator = Evaluator(
                data=data_loader,
                granularity=configs.granularity,
                start_periods_eval=configs.evaluation_start,
                end_periods_eval=configs.evaluation_end,
                forecasting_horizon=configs.forecasting_horizons[0],
                granularity_eval=configs.column_historical_forecast,
                is_sell_in_model=configs.is_sell_in_model,
                is_weekly_forecast=configs.is_weekly_forecast
            )
            evaluation = evaluator.eval(
                res,
                data_loader.df_cpg_trade_channel if configs.is_sell_in_model else None,
                data_loader.df_materials_master_data if configs.is_sell_in_model else None,
            )

            # Assumption: should not be any data on F15 according to Valerie
            if configs.is_sell_in_model:
                evaluation = evaluation[evaluation[n.FIELD_PLANT_ID] != 'F015']

            evaluator.report(evaluation, configs.granularity_evaluator.values())
            if configs.is_sell_in_model:
                evaluator.detailed_report(evaluation, configs.granularity_evaluator.values())

    # Save model
    if configs.train_model:
        mod.save_model(
            forecasting_horizon=configs.forecasting_horizons[0],
            logger=Logger.logger,
            configs=configs
        )


if __name__ == '__main__':
    START = time.time()

    configs = None
    if len(sys.argv) > 1:
        configs = Configs(sys.argv[1])

    main(configs)
    END = time.time()
    Logger.info('Script completed in', '%i seconds' % str(END - START), __file__)
