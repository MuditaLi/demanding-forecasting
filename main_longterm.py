"""
The entry point of the long-term demand forecasting model.
"""
import sys
import time
from configs.configs import Configs
import pandas as pd
from preprocessor.data_loader import DataLoader
from util.logger import Logger
from dcd_forecast_model.long_term_trainer import LongTermPredictor


def main(configs: Configs=None, data_loader: DataLoader=None):
    """
    Main function of the demand forecasting model

    Usage example:
        > python carlsberg/main_longterm.py
    """
    # Load data
    if configs is None:
        configs = Configs('longterm_default.yaml')

    # data_loader = None
    if data_loader is None:
        data_loader = DataLoader(configs)
        data_loader.load_data(configs.main_file_name)

    longtrainer = LongTermPredictor(
        start_train=configs.train_start,
        end_train=configs.train_end,
        end_test=configs.test_end,
        start_test=configs.test_start,
        forecasting_horizons=configs.forecasting_horizons
    )
    temp = longtrainer.fit_and_predict(data_loader)
    time = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M')
    temp.to_csv("outputs/longterm_model_"+time+".csv", index=False)


if __name__ == '__main__':
    START = time.time()

    main()
    END = time.time()
    Logger.info('Script completed in', '%i seconds' % (END - START), __file__)
