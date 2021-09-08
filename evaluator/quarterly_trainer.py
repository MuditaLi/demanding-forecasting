"""
Script to assess the performance of our model whenever it is retrained on a quarterly basis (back-testing on 2018).
"""
import os
import pickle
import pandas as pd
from configs.configs import Configs
from .evaluation import Evaluator
from dcd_forecast_model.trainer import Predictor, Trainer
import preprocessor.names as n
from preprocessor.data_loader import DataLoader
from dcd_forecast_model.model import ModelFactory
from util.logger import Logger


def quarterly_trained_model(configs: Configs, data_loader: DataLoader, quarter: str):

    if configs.train_model:
        tr = Trainer(
            start_train=configs.train_start,
            end_train=configs.train_end,
            granularity=configs.granularity,
            forecasting_horizons=configs.forecasting_horizons,
            feature_controller=configs.features,
            is_weekly_forecast=configs.is_weekly_forecast,
            is_sell_in_model=configs.is_sell_in_model,
            use_light_regressor=configs.use_light_regressor,
        )
        mod, train = tr.fit(data_loader, parameters=configs.model_parameters)
    else:
        mod = ModelFactory(use_light_regressor=configs.use_light_regressor)
        mod.load_model(file_name=configs.trained_model_file,
                       forecasting_horizon=configs.forecasting_horizons[0],
                       is_sell_in_model=configs.is_sell_in_model,
                       quarter=quarter)

    predictor = Predictor(
            start_test=configs.test_start,
            end_test=configs.test_end,
            model=mod,
            granularity=configs.granularity,
            forecasting_horizons=configs.forecasting_horizons
        )
    res = predictor.predict(data_loader)

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
    evaluation = evaluator.eval(res, data_loader.df_cpg_trade_channel if configs.is_sell_in_model else None)
    evaluation = evaluation[evaluation[n.FIELD_PLANT_ID] != 'F015']
    evaluator.report(evaluation, configs.granularity_evaluator.values())

    # Save model
    mod.save_model(
        forecasting_horizon=configs.forecasting_horizons[0],
        is_sell_in_model=configs.is_sell_in_model,
        logger=Logger.logger,
        configs=configs,
        quarter=quarter
    )

    # Save evaluation table
    file_path = os.path.join(os.getcwd(), 'predictions', 'predictions_%s_w%d.pkl' %
                             (quarter.lower(), configs.forecasting_horizons[0]))

    with open(os.path.join(os.getcwd(), file_path), 'wb') as f:
        pickle.dump(evaluation, f)
    Logger.info('Predictions saved in file', file_path, 'QuarterlyTrainer')


if __name__ == '__main__':

    evaluations = {
        'Q1': [201801, 201813],
        'Q2': [201814, 201826],
        'Q3': [201827, 201839],
        'Q4': [201840, 201852],
    }

    configs = Configs('default_france.yaml')
    data_loader = DataLoader(configs)
    data_loader.load_data(configs.main_file_name)

    for quarter, eval_periods in evaluations.items():
        setattr(configs, 'evaluation_start', eval_periods[0])
        setattr(configs, 'evaluation_end', eval_periods[1])

        # Update train_end attribute to get the largest possible scope of data
        configs.update_train_test_windows(force_train_dates_from_evaluation=True)

        # Train | Load quarterly trained model to get a better prediction
        quarterly_trained_model(configs, data_loader, quarter)

    # Evaluation
    results = list()
    for quarter in evaluations:
        file_name = 'predictions_%s_w%d.pkl' % (quarter.lower(), configs.forecasting_horizons[0])
        with open(os.path.join(os.getcwd(), 'predictions', file_name), 'rb') as f:
            results.append(pickle.load(f))

    results = pd.concat(results, axis=0, sort=False)

    # Merge with evaluator df
    reference = pd.read_csv(os.path.join(os.getcwd(), 'predictions', 'evaluationw4.csv'), usecols=configs.granularity)
    results = reference.merge(results, on=configs.granularity, how='inner')
    evaluator = Evaluator(
        data=data_loader,
        granularity=configs.granularity,
        start_periods_eval=min(sum(evaluations.values(), [])),
        end_periods_eval=max(sum(evaluations.values(), [])),
        forecasting_horizon=configs.forecasting_horizons[0],
        granularity_eval=configs.column_historical_forecast,
        is_sell_in_model=configs.is_sell_in_model,
        is_weekly_forecast=configs.is_weekly_forecast
    )
    evaluation = evaluator.eval(results, data_loader.df_cpg_trade_channel if configs.is_sell_in_model else None)
    evaluator.report(evaluation, configs.granularity_evaluator.values())
