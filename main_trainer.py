"""
Script to train multiple models
$ python main_trainer.py default_trainer_france.yaml
"""
import time
import sys
from configs.configs import ConfigsTrainer
from main import main
from preprocessor.data_loader import DataLoader
from util.logger import Logger


def train_weekly_models(configs_trainer: ConfigsTrainer):

    configs = configs_trainer.configs_individual_model
    data_loader = DataLoader(configs, configs.features)
    data_loader.load_data(configs.main_file_name)

    original_train_end = configs.train_end
    for horizon in sorted(configs_trainer.forecasting_horizons_trainer):
        start = time.time()

        # Update configs attributes (e.g. train_end for given horizon)
        setattr(configs, 'forecasting_horizons', [horizon])
        train_end = configs.update_train_end_to_meet_maximum_date_to_predict(
            original_train_end, configs_trainer.train_date_to_predict_max, horizon)

        if train_end != configs.train_end:
            setattr(configs, 'train_end', train_end)
            configs.update_train_test_windows()

        # Run main model
        main(configs, data_loader)
        # print(horizon, configs.train_start, configs.train_end)

        time_granularity, model_type = ('in', 'w') if configs.is_sell_in_model else ('out', 'm')
        Logger.info('%s-%d sell-%s model trained in' % (model_type, horizon, time_granularity),
                    '%d seconds' % round(time.time() - start, 1), __file__)


if __name__ == '__main__':

    # sys.argv[1] = 'default_trainer_france.yaml'
    configs_trainer = ConfigsTrainer(sys.argv[1])  # Trainer configs object
    train_weekly_models(configs_trainer)
