import numpy as np
import pandas as pd
from configs.feature_controller import FeatureController
import preprocessor.names as n
from preprocessor.data_loader import DataLoader
from util.misc import create_list_period, get_all_datehorizons
from util.splits import get_columns_scenarios
from util.logger import Logger
from .model import ModelFactory
from . import FeatureEngineeringSellIn, FeatureEngineeringSellOut


class Config:

    NOT_FEATURES = [
        n.FIELD_LEAD_SKU_ID,
        n.FIELD_PLANT_ID,
        n.FIELD_SKU_EAN,
        n.FIELD_CUSTOMER,
        n.FIELD_STORE_NAME,
        n.FIELD_STORE_ID,
        n.FIELD_CUSTOMER_GROUP,
        'date_when_predicting',
        'date_to_predict',
        n.FIELD_LABEL,
        n.FIELD_QTY_SOLD_SU,
        n.FIELD_PREDICTION
    ]

    def __init__(self, granularity: list, forecasting_horizons: list, feature_controller: FeatureController=None,
                 is_weekly_forecast: bool=True, is_sell_in_model: bool=True, use_light_regressor: bool=False):
        """
        Args:
            is_weekly_forecast (bool): Whenever False, forecast at weekly level. Otherwise, forecast at monthly level
            is_sell_in_model (bool): Boolean to indicate whether we want to train the sell_in or the sell_out model
        """

        self.granularity = granularity
        self.feature_controller = feature_controller
        self.forecasting_horizons = forecasting_horizons
        self.is_weekly_forecast = is_weekly_forecast
        self.is_sell_in_model = is_sell_in_model
        self.use_light_regressor = use_light_regressor

    @staticmethod
    def find_relevant_feature_columns(df):
        return df.columns.difference(Config.NOT_FEATURES)


class Trainer(Config):

    def __init__(self, start_train, end_train, custom_loss_function=None, **kwargs):
        """
        Model trainer object.
        Args:
            custom_loss_function(function or None): If not None, we use this loss function to train the model.
        """

        super().__init__(**kwargs)

        self.custom_loss_function = custom_loss_function
        self.start_train = start_train
        self.end_train = end_train

    def build_train_set(self, data: DataLoader):
        """
        Build the training set, the labels

        Args:
            data (DataLoader): data object containing all relevant tables

        Returns:
            train (pd.DataFrame): Train set including target labels
            x_train : train set in numpy array
            y_train: labels
        """
        dwp = create_list_period(self.start_train, self.end_train, self.is_weekly_forecast)
        dtp = get_all_datehorizons(dwp, self.forecasting_horizons, self.is_weekly_forecast)
        Logger.info('Scope data used to train model (dates_to_predict)',
                    '[%d, %d]' % (min(dtp), max(dtp)), self.__class__.__name__)

        if self.is_sell_in_model:
            feature_creator = FeatureEngineeringSellIn(
                is_weekly_forecast=self.is_weekly_forecast,
                granularity=self.granularity,
                feature_controller=self.feature_controller
            )
        else:
            feature_creator = FeatureEngineeringSellOut(
                is_weekly_forecast=self.is_weekly_forecast,
                granularity=self.granularity,
                feature_controller=self.feature_controller
            )

        # building training set and labels
        train = feature_creator.build_train(data, dwp, dtp)

        x_train = train[Config.find_relevant_feature_columns(train)]
        y_train = train[n.FIELD_LABEL]

        return feature_creator, train, x_train, y_train

    def fit(self, data: DataLoader, regressor=None, parameters=None):
        """
        Build the training set, the labels and train th model

        Args:
            data (DataLoader): data object containing all relevant tables

        Returns:
            model (ModelFactory): Trained model
            train (pd.DataFrame): Train set including target labels
        """

        feature_creator, train, x_train, y_train = self.build_train_set(data)

        # train model
        model = ModelFactory(
            regressor=regressor,
            parameters=parameters,
            feature_creator=feature_creator,
            use_light_regressor=self.use_light_regressor,
            custom_loss_function=self.custom_loss_function
        )

        # self.feature_creator = feature_creator
        model.train_model(x_train, y_train)

        return model, train


class Predictor(Config):

    def __init__(self, start_test, end_test, model, **kwargs):

        super().__init__(**kwargs)

        self.start_test = start_test
        self.end_test = end_test
        self.model = model

    def predict(self, data: DataLoader):
        """
        Build features set, target labels and predict label using attribute model
        Args:
            data (DataLoader): data object containing all relevant tables

        Returns:
            test (pd.Dataframe): test set with both target labels and predicted labels
        """
        feature_creator = self.model.feature_creator

        dwp_test = create_list_period(self.start_test, self.end_test, feature_creator.is_weekly_forecast)
        dtp_test = get_all_datehorizons(dwp_test, self.forecasting_horizons, feature_creator.is_weekly_forecast)

        test = feature_creator.build_predict(data, dwp_test, dtp_test)
        x_test = test[Config.find_relevant_feature_columns(test)]

        test[n.FIELD_PREDICTION] = self.model.predict_model(x_test)
        return test

    def predict_base_line(self, test: pd.DataFrame):
        """
        Predict volumes without promotion information to predict the baseline volume & promotion uplift
        estimated by M.L. model.

        Args:
            test (pd.DataFrame): test set including predictions from ML model with actual promotion plan known at time
            of predictions.

        Returns:
            test (pd.Dataframe): test set with both target labels and predicted labels
        """
        x_test = test[Config.find_relevant_feature_columns(test)].copy()

        promo_feature_names = [
            'delta_prom',
            'mecanique_Aucune',
            'mecanique_LV 2eme a 50%',
            'mecanique_LV 3 pour 2',
            'mecanique_RI %',
            'mecanique_Ticket/Cagnottage %',
            'mecanique_autre_meca',
            'natreg_Nationale',
            'natreg_Regionale',
            'prev_cm_hl',
            'promo_in_brand',
            'promo_in_subbrand',
            'promo_in_unitperpack',
            'type_d\'offre_Offre Complementaire',
            'type_d\'offre_Prospectus',
            'type_d\'offre_Saisonnier',
            'type_d\'offre_autre_type',
            'unit_per_pack',
            'valeur_mecanique',
            'promotion_uplift'
        ]

        x_test[x_test.columns.intersection(promo_feature_names)] = np.nan
        # x_test['promotion_uplift'] = 0
        x_test['ml_baseline_wo_promo'] = self.model.predict_model(x_test, ignored_columns=get_columns_scenarios(x_test))

        test['ml_baseline_wo_promo'] = x_test['ml_baseline_wo_promo']
        test['ml_promo_uplift'] = test[n.FIELD_PREDICTION] - test['ml_baseline_wo_promo']

        # test.groupby('date_to_predict')[['predictions', 'ml_promo_uplift', 'ml_baseline_wo_promo']].sum().transpose()
        return test

    def run_weather_scenarios(self, test: pd.DataFrame, data: DataLoader, degrees_celsius: [float]=None,
                              use_existing_year_data: [int]=None):
        """
        Run weather scenarios. Either add constant delta of temperature throughout the period.
        Ideally, please use it with models that only include temperature as weather features.

        Args:
            data (DataLoader): all relevant data objects
            test (pd.DataFrame): test set including predictions from ML model with temperature data
            degrees_celsius (list of float): list of temperature deltas to apply (in celsius)
            use_existing_year_data (list of int): list of year to apply that year's weather to our model.
            If we do so, we assume that all weeks in scope belong to the same year.

            Please use either degrees_celsius or use_existing_year_data but not both.
        """
        x_test = test[Config.find_relevant_feature_columns(test)].copy()

        # Copy original temperature features
        cols = list(filter(lambda x: x.startswith('apparent_temperature_mean_') and not x.endswith('_org'),
                           x_test.columns))
        x_test[list(map(lambda x: x + '_org', cols))] = x_test[cols].copy()

        # print('Normal Features')
        # print(x_test[cols].drop_duplicates().mean())

        # Scenario with fixed delta of temperature
        if degrees_celsius is not None:
            for delta in degrees_celsius:
                # Updated data frame of features
                tmp_test = self.model.feature_creator.replace_weather_features_in_feature_table(
                    df_test=test, data=data, delta_celsius=delta)

                # Store week temperature and predictions
                test['_'.join(['temperature', 'w0', 'celsius_%d' % delta])] = test['apparent_temperature_mean_week0']
                test['_'.join([n.FIELD_PREDICTION, 'celsius_%d' % delta])] = self.model.predict_model(
                    tmp_test[Config.find_relevant_feature_columns(tmp_test)],
                    ignored_columns=get_columns_scenarios(tmp_test)
                )
                test[cols] = test[list(map(lambda x: x + '_org', cols))].copy()

        # Scenarios with temperature of other years
        if use_existing_year_data is not None:
            for new_year in use_existing_year_data:
                # Updated data frame of features
                tmp_test = self.model.feature_creator.replace_weather_features_in_feature_table(
                    df_test=test, data=data, use_existing_year=new_year)

                # Store week's temperature and predictions
                test['_'.join(['temperature', 'w0', 'weather_%d' % new_year])] = \
                    tmp_test['apparent_temperature_mean_week0']

                test['_'.join([n.FIELD_PREDICTION, 'weather_%d' % new_year])] = self.model.predict_model(
                    tmp_test[Config.find_relevant_feature_columns(tmp_test)],
                    ignored_columns=get_columns_scenarios(tmp_test)
                )

        # print(test[[col for col in test.columns if col.startswith(n.FIELD_PREDICTION)]].sum())

        # Write weather features from forecast back
        test[cols] = test[list(map(lambda x: x + '_org', cols))].copy()
        test.drop(columns=list(map(lambda x: x + '_org', cols)), inplace=True)

        return test
