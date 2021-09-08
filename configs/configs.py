import os
from util.paths import PATH_CONFIG
import util.input_output as io
from util.misc import substract_period
from util.logger import Logger
from configs.feature_controller import FeatureController


class Configs:

    country = None
    evaluation_start = None
    evaluation_end = None
    file_name_historical_forecasts = ''
    features = None
    forecasting_horizons = list()
    granularity = list()
    granularity_evaluator = list()
    is_sell_in_model = True
    is_weekly_forecast = True
    main_data_version = ''  # Which data version should we use - folder name containing the right data
    main_file_name = ''
    model_parameters = None
    column_historical_forecast = None
    train_model = True  # Do I want to train the model or not?
    trained_model_file = None  # Name of pickle file containing all information regarding the model of interest
    train_start = None
    train_end = None
    test_start = None
    test_end = None
    use_light_regressor = False

    def __init__(self, config_file_name: str=None, configs: dict=None, is_sell_in_model: bool=None):
        """
        Create configs element from file or from a dictionary of values

        Args:
             config_file_name (str): name of the configs file
             configs (dict): settings as a dictionary
             is_sell_in_model (bool): boolean to force which model to select (useful for the future forecast)
        """

        if config_file_name is not None:
            configs = io.read_yaml(PATH_CONFIG, config_file_name)

            if is_sell_in_model is None:
                self.is_sell_in_model = configs['model'] == 'sell_in'

            configs = configs['sell_in' if self.is_sell_in_model else 'sell_out']
            Logger.info('Loaded configs from file', os.path.join(PATH_CONFIG, config_file_name), self.__class__.__name__)

        for name, value in configs.items():
            if not name.endswith('_parameters'):
                self.__setattr__(name, value)

        # Add model parameters from configs file
        model_parameters = configs.get('_'.join(['lightgbm' if self.use_light_regressor else 'xgboost', 'parameters']))
        self.__setattr__('model_parameters', model_parameters)

        # Add features controller object to configs (ability to switch on & off features in model trainer)
        if self.features is not None:
            self.features = FeatureController(self.is_sell_in_model, self.features)

        if self.forecasting_horizons:
            self.update_train_test_windows()

    def to_dict(self):
        """ Method to convert all configs attributes to dictionaries (saved trained model) """
        return {k: v.__dict__ if isinstance(v, FeatureController) else v for k, v in self.__dict__.items()}

    def update_train_test_windows(self, force_train_dates_from_evaluation: bool=False):
        """ Logic to determine dates when predicting for test and train based """
        max_period = 52 if self.is_weekly_forecast else 12
        horizon = self.forecasting_horizons[0]

        if self.evaluation_start is not None and self.evaluation_end is not None:
            self.test_end = substract_period(self.evaluation_end, horizon, highest_period=max_period)
            self.test_start = substract_period(self.evaluation_start, horizon, highest_period=max_period)

            if self.train_end >= self.test_start or force_train_dates_from_evaluation:
                self.train_end = substract_period(self.test_start, 1, highest_period=max_period)

    def update_train_end_to_meet_maximum_date_to_predict(self, train_end_org: int,
                                                         date_to_predict_max: int, forecasting_horizon: int):
        """
        Method to automatically adapt the test and train periods to be consistent with evaluation periods
        whenever specified.
        """
        max_period = 52 if self.is_weekly_forecast else 12
        train_end = substract_period(date_to_predict_max, forecasting_horizon, highest_period=max_period)
        if train_end_org is None or train_end < train_end_org:
            return train_end

        return train_end_org


class ConfigsTrainer:

    forecasting_horizons_trainer = [1]  # All horizons for which we want to train a model
    train_date_when_predicing_min = None
    train_date_to_predict_max = None

    def __init__(self, config_file_name: str):

        """
        Config file for main trainer factory (Scheduler to train multiple models in a row
        i.e. for a specified set of horizons)

        To ensure that we store the model settings only in one place, the configs Trainer will take the default values
        given in the model configs file (that must be specified in file_name_model_configs arguments in trainer configs
        yaml file). Attributes that are also specified in the trainer configs will replace the ones in the model configs

        Args:
            config_file_name (str): name of yaml file in configs folder containing the configurations
            if evaluation_start is None, model is trained on all data.
            e.g. config_file_name = "default_trainer_france.yaml"

        """
        configs_trainer = io.read_yaml(PATH_CONFIG, config_file_name)
        configs_model = configs_trainer[configs_trainer['model']]

        # Add trainer configs attributes
        horizons = configs_trainer['forecasting_horizons_trainer']
        self.forecasting_horizons_trainer = range(horizons['smallest_horizon'], horizons['largest_horizon'] + 1)

        for name, value in configs_trainer.items():
            if name in ['train_date_when_predicing_min', 'train_date_to_predict_max']:
                self.__setattr__(name, value)

        # Initiate individual model configs object (replace attributes that were specified in configs_model).
        configs = io.read_yaml(PATH_CONFIG, configs_trainer['file_name_model_configs'])
        configs = configs[configs_trainer['model']]
        Logger.info('Loaded model configs from file',
                    os.path.join(PATH_CONFIG, configs_trainer['file_name_model_configs']), self.__class__.__name__)
        configs.update(configs_model)

        def update_train_scope(attr, limit, fct):
            if configs.get(attr) is not None and limit in vars(self):
                date = fct(configs.get(attr), self.__getattribute__(limit))
                configs.update({attr: date})

        update_train_scope('train_start', 'train_date_when_predicting_min', max)
        update_train_scope('train_end', 'train_date_to_predict_max', min)

        self.configs_individual_model = Configs(configs={k: v for k, v in configs.items()
                                                         if k in Configs.__dict__.keys()})

        # Update maximum date to predict train to ensure that we don't overlap with the evaluation period
        if self.configs_individual_model.evaluation_start is not None and self.train_date_to_predict_max is not None:
            max_date_to_predict = substract_period(
                self.configs_individual_model.evaluation_start, 1,
                highest_period=52 if self.configs_individual_model.is_weekly_forecast else 12
            )
            self.train_date_to_predict_max = min(self.train_date_to_predict_max, max_date_to_predict)

        Logger.info('Loaded trainer configs from file',
                    os.path.join(PATH_CONFIG, config_file_name), self.__class__.__name__)


class ConfigsFuturePredictor:

    date_when_predicting = None
    first_date_to_predict = None
    last_date_to_predict = None
    file_name_configs_model = None
    file_name_predictions_list = None
    scenario_promo = False
    scenario_weather = dict()  # specify parameters for the weather scenario

    def __init__(self, config_file_name):
        """
        Configs object for future forecast script, which provides forecast for multiple horizons
        Format week = YYYYWW as an integer or month = YYYYMM as an integer

        Args:
            config_file_name (str): name of the Future forecast configs file

        Configs file must contain the following information:
            - date_when_predicting (int): week (or month) number at which you are doing the forecast
            - first_date_to_predict (int): week (or month) number of first week to predict
            - last_date_to_predict (int): week (or month) number of last week to predict
            - file_name_predictions_list (str): list of all SKU, CPG, Plant, Wk or Mo to predict.
            Allows user to deal with phase in and phase out of SKUs in system
            - file_name_configs_model (str): name of a single model configs file
        """
        configs = io.read_yaml(PATH_CONFIG, config_file_name)
        Logger.info('Loaded future forecasts configs from file',
                    os.path.join(PATH_CONFIG, config_file_name), self.__class__.__name__)

        self.is_sell_in_model = configs['model'] == 'sell_in'
        for name, value in configs[configs['model']].items():
            self.__setattr__(name, value)
