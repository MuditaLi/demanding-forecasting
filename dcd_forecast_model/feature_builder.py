import pandas as pd
from functools import reduce
from configs.feature_controller import FeatureController


class FeatureEngineering:

    pipelines = dict()  # By default pipelines is an empty dictionary

    def __init__(self, granularity, feature_controller: FeatureController, is_weekly_forecast: bool=True):
        """
        All common functions used by sell-in & sell-out features engineering objects

        Args:
            granularity (list): granularity of base prediction
            feature_controller (FeatureController): controller to indicate which features to turn on
            is_weekly_forecast (bool): Is the base prediction at weekly level?
        """
        self.pipelines = FeatureEngineering.pipelines  # Needed for pickling pipelines in model
        self.feature_controller = feature_controller
        self.granularity = granularity
        self.is_weekly_forecast = is_weekly_forecast

    @staticmethod
    def consolidate_features(frame: [pd.DataFrame]):
        """
        Args:
            frame (list of pd.DataFrame):
        Returns:
            df_final (pd.DataFrame): Consolidated feature data set.
        """
        return reduce(
            lambda left, right: pd.merge(left, right, on=list(set(left.columns) & set(right.columns)), how='left'),
            frame
        )
