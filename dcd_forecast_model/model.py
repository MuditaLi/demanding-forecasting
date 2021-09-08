"""
This module implements a versatile ModelFactory class to store the demand forecast model.
"""
import os
import json
import pandas as pd
import pickle
from util.logger import Logger
from configs.configs import Configs
from util.paths import PATH_MODELS
# from sklearn.ensemble import RandomForestClassifier
import lightgbm
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# def custom_carlsberg_train(y_true, y_pred):
#     return "custom_carlsberg_objective", carlsberg_score(y_true, y_pred), False
#
#
# def custom_carlsberg_objective(y_true, y_pred):
#     # residual = (y_true - y_pred).astype("float")
#     grad = np.where(y_pred == 0, 0.1, - y_true * np.sign(y_true - y_pred) / y_pred ** 2)
#     hess = np.where(y_pred == 0, 0.1, 2 * y_true * np.sign(y_true - y_pred) / y_pred ** 3)
#     return grad, hess


class ModelFactory:
    """
    A versatile model class for regression purpose
    Parameters are set to default values but needs to be adjusted when the model is tested on a new markt, or when
    new features/data are included. In particular number of trees and depth need to be changed
    """

    DEFAULT_XGBOOST_PARAMETERS = {
        'max_depth': 15,  # 25
        'n_estimators': 70,
        'learning_rate': 0.1,
        'n_jobs': 12,
        'verbosity': 2
    }

    DEFAULT_LIGHT_XGBOOST_PARAMETERS = {
        'boosting_type': 'gbdt',
        'max_depth': 25,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample_for_bin': 200000,
        'application': 'regression',
        'n_jobs': 12,
        'silent': False
    }

    covariates = None  # List of names of features used by model

    # DEFAULT_LIGHT_XGBOOST_PARAMETERS = {
    #     'boosting_type': 'gbdt',
    #     'num_leaves': 31,
    #     'max_depth': 3,
    #     'learning_rate': 0.05,
    #     'n_estimators': 200,
    #     'subsample_for_bin': 200000,
    #     'application': 'regression',
    #     'metric': None,
    #     'class_weight': None,
    #     'min_split_gain': 0.0,
    #     'min_child_weight': 0.001,
    #     'min_data_in_leaf': 20,
    #     'subsample': 1.0,
    #     'subsample_freq': 1,
    #     'feature_fraction': 1.0,
    #     'reg_alpha': 0.0,
    #     'reg_lambda': 0.0,
    #     'random_state': None,
    #     'n_jobs': 12,
    #     'silent': False
    # }

    def __init__(self, regressor=None, parameters=None, feature_creator=None, use_light_regressor: bool=True,
                 custom_loss_function=None):
        """
        Create an empty model

        Args:
            regr: a classifier (possibly from sklearn model with at least "fit" and "predict_proba" methods.
            If None, a xgboost model is used)

            parameters (dict): a dictionary of parameters that can be passed to the regressor

            use_light_regressor (bool): switch between default models (light vs normal XGBoost)
        """
        self.feature_creator = feature_creator

        if parameters is None:
            parameters = ModelFactory.DEFAULT_LIGHT_XGBOOST_PARAMETERS \
                if use_light_regressor else ModelFactory.DEFAULT_XGBOOST_PARAMETERS

            if custom_loss_function is not None:
                parameters.update({'objective': custom_loss_function})
                Logger.warning('Model trained with custom loss function',
                               custom_loss_function.__name__,
                               self.__class__.__name__)

        self.regr_parameters = parameters

        if regressor is None:
            regressor = lightgbm.LGBMRegressor if use_light_regressor else xgb.XGBRegressor
            Logger.info('Model set to', regressor.__name__, self.__class__.__name__)

        self.regr = regressor(**self.regr_parameters)

    def save_model(self, forecasting_horizon: int, configs: Configs=None, logger: Logger=None, quarter: str=None):
        """
        Save the ModelFactory instance to a pickle with its metadata

        Args:
            forecasting_horizon (int): in weeks or months
            configs (Configs): configs element used to train the model (including the country name)
            logger (Logger): logger element
            quarter (str): Leave to None by default. Please specify a value whenever using the quarterly trainer to
            assess the true performance of the model on back-test data.
        """
        model_type, horizon = ('in', 'w') if configs.is_sell_in_model else ('out', 'm')
        model_horizon = horizon + str(forecasting_horizon)

        time = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M')
        path = os.path.join(PATH_MODELS, configs.country, 'sell_' + model_type, model_horizon)

        if quarter is not None:
            path = os.path.join(path, quarter.lower())

        os.makedirs(path, exist_ok=True)

        # Save model
        file_name = '_'.join([time, 'forecast_model.pkl'])
        with open(os.path.join(path, file_name), 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        Logger.info('Model saved in file', os.path.join(path, file_name), self.__class__.__name__)

        # Save configs element
        if configs is not None:
            file_name = '_'.join([time, 'configs.json'])
            with open(os.path.join(path, file_name), 'w') as f:
                json.dump(configs.to_dict(), f)

        # Save logs of run
        if logger is not None:
            file_name = '_'.join([time, os.path.basename(Logger.file_handler_run.baseFilename)])
            with open(os.path.join(path, file_name), 'w') as f:
                with open(Logger.file_handler_run.baseFilename) as log_file:
                    f.write(log_file.read())

    def load_model(self, forecasting_horizon: int, is_sell_in_model: bool, country: str='',
                   file_name: str=None, quarter: str=None):
        """
        Load a previously trained model

        Args:
            country (str): which country folder should we save the model in
            file_name (str): name of pickle file containing the saved model
            forecasting_horizon (int): forecasting horizon in weeks or months
            is_sell_in_model (bool): Is it a sell-in or sell-out model?
            quarter (str): parameter to quarterly retrain a model
        """
        model_type, horizon = ('in', 'w') if is_sell_in_model else ('out', 'm')
        model_horizon = horizon + str(forecasting_horizon)
        path = os.path.join(PATH_MODELS, country, 'sell_' + model_type, model_horizon)

        if quarter is not None:
            path = os.path.join(path, quarter.lower())

        if file_name is None:
            files = [f for f in os.listdir(path) if f.endswith('forecast_model.pkl')]
            files = map(lambda f: pd.to_datetime(''.join(f.split('_')[:2]), format='%Y%m%d%H%M'), files)
            file_name = sorted(files, reverse=True)[0].strftime('%Y%m%d_%H%M_forecast_model.pkl')

        with open(os.path.join(path, file_name), 'rb') as f:
            loaded_model = pickle.load(f)

        self.__dict__.update(loaded_model)
        Logger.info('Model loaded from file', os.path.join(path, file_name), self.__class__.__name__)
        Logger.info('Model switched to', '\n' + self.regr.__repr__(), self.__class__.__name__)

    def train_model(self, x, y):
        """
        Fit the classifier to the data

        :param x: a pandas dataframe or numpy array with the training set
        :param y: the labels for the classifier
        :param x: name of columns in
        """
        if isinstance(x, pd.DataFrame):
            self.covariates = list(x.columns)
            x = x.values

        self.regr.fit(x, y)
        Logger.info(f"{self.regr.__str__()} fitted", class_name=self.__class__.__name__)

    @staticmethod
    def hyper_parameter_opt(x, y, parameter_grid):
        """
        Fit the  best hyper-parameters for the data

        :param x: a pandas dataframe or numpy array with the training set
        :param y: the labels for the classifier
        :param parameter_grid: grid of parameters to explore
        """
        # setup regressor
        xgb_model = xgb.XGBRegressor(n_jobs=1)

        if parameter_grid is None:
            parameter_grid = {'max_depth': [31],
                              'n_estimators': [70],
                              'learning_rate': [0.1],
                              'min_child_weight': [1],
                              'gamma': [0.1],
                              'subsample': [1],
                              'colsample_bytree': [1],
                              'reg_alpha': [0]
                              }

        # perform a grid search
        tweaked_model = GridSearchCV(
            xgb_model,
            parameter_grid,
            cv=6,
            verbose=1,
            n_jobs=7,
            scoring='neg_mean_absolute_error'
        )

        tweaked_model.fit(x, y)

        # summarize results
        print("Best: %f using %s" % (tweaked_model.best_score_, tweaked_model.best_params_))
        return tweaked_model

    def predict_model(self, x, ignored_columns: set=set()):
        """
        Predict the demand
        Args:
            x (pd.DataFrame or np.Array): table with the features of the customers
            ignored_columns (set): set of column names that you don't want to check that they are used or not

        Returns:
            Predicted shipments / sell-in sales volumes in hl
        """
        if self.covariates is not None:

            features_missing = list(set(self.covariates).difference(x.columns))
            if features_missing:
                Logger.debug('Following features are missing in features table', '\n- '.join([''] + features_missing),
                             class_name=self.__class__.__name__)
                return

            # Highlight feature that were created, but not used by model to predict (removing scenario output columns)
            features_ignored = list(set(x.columns).difference(self.covariates).difference(ignored_columns))

            if features_ignored:
                Logger.warning('Following features not used by model', '\n- '.join([''] + features_ignored),
                               class_name=self.__class__.__name__)

            x = x[self.covariates]

        if isinstance(x, pd.DataFrame):
            x = x.values

        pred = self.regr.predict(x)
        Logger.info('Prediction of volume demand', 'done', class_name=self.__class__.__name__)

        return pred

    def summary_feature_importance_table(self):
        """ """
        if isinstance(self.regr, list):
            return pd.DataFrame(list(zip(self.covariates, self.regr.feature_importances_)),
                                columns=['feature', 'fscore']).round(3)
