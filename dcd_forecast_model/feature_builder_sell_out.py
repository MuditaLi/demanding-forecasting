import pandas as pd
import features
import util.misc as misc
import preprocessor.names as n
from dcd_forecast_model.feature_builder import FeatureEngineering
from preprocessor.data_loader import DataLoader
from util.logger import Logger


class FeatureEngineeringSellOut(FeatureEngineering):
    """
    This class loads the data, build features and stack them to build the training or predict set
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_all_sell_out_features(self,
                                    data: DataLoader,
                                    dates_when_predicting: list,
                                    dates_to_predict: list,
                                    is_training: bool=True) -> pd.DataFrame:

        feature_tables = list()
        df_materials = misc.convert_categorical(data.df_ean_master_data)

        df_sell_out = data.df_sell_out.groupby(self.granularity)[n.FIELD_LABEL].sum().unstack(
            n.FIELD_YEARMONTH).fillna(0)

        def add_features_to_frame(message, df: [pd.DataFrame], name=self.__class__.__name__):
            """ helper function to append latest features data frame & print custom log"""
            feature_tables.append(df)
            Logger.info(message, 'done', name)

        if self.feature_controller.sales_customer_store_consumer_unit_last_year_rolling:
            # Reference data frame giving all relevant plant / cpg / ean combinations. Always first table to build.
            df1 = features.sales.features_amount_sales(
                df_sell_out, dates_when_predicting, dates_to_predict,
                timelag=52 if self.is_weekly_forecast else 12,
                is_weekly=self.is_weekly_forecast
            )
            add_features_to_frame('Build sales features', df1)

        # Actual volumes sold (dependant variable)
        df2 = features.sales.features_labels(df_sell_out, dates_when_predicting, dates_to_predict)
        add_features_to_frame('Add target volume', df2)

        # EAN characteristics
        if self.feature_controller.consumer_unit_characteristics:
            if is_training:
                df3, materials_pipeline = features.materials.features_materials(df_materials, is_training)
                self.pipelines['materials'] = materials_pipeline
            else:
                df3, _ = features.materials.features_materials(df_materials, is_training, self.pipelines['materials'])
            add_features_to_frame('Build SKU describing features', df3)

        # Business activity features
        if self.feature_controller.sales_customer_store_last_year_rolling:
            df4 = features.sales.create_rolling_sales_volume(data.df_sell_out,
                                                             [n.FIELD_CUSTOMER, n.FIELD_STORE_ID], n.FIELD_YEARMONTH,
                                                             'store_total_sales_LY_rw')
            add_features_to_frame('Build business activity features (1/5)', df4)

        if self.feature_controller.sales_customer_last_year_rolling:
            df5 = features.sales.create_rolling_sales_volume(data.df_sell_out,
                                                             [n.FIELD_CUSTOMER], n.FIELD_YEARMONTH,
                                                             'customer_total_sales_LY_rw')
            add_features_to_frame('Build business activity features (2/5)', df5)

        if self.feature_controller.sales_customer_consumer_unit_last_year_rolling:
            df6 = features.sales.create_rolling_sales_volume(data.df_sell_out,
                                                             [n.FIELD_CUSTOMER, n.FIELD_SKU_EAN], n.FIELD_YEARMONTH,
                                                             'customer_ean_total_sales_LY_rw')
            add_features_to_frame('Build business activity features (3/5)', df6)

        if self.feature_controller.sales_customer_brand_last_year_rolling:
            df7 = features.sales.create_rolling_sales_volume_per_category(data.df_sell_out, data.df_ean_master_data,
                                                                          [n.FIELD_CUSTOMER, n.FIELD_SKU_BRAND],
                                                                          n.FIELD_YEARMONTH,
                                                                          'cust_brand_total_sales_LY_rw')
            add_features_to_frame('Build business activity features (4/5)', df7)

        # Seasonality features
        if self.feature_controller.seasonality:
            df8 = features.features_date.create_seasonality_features(dates_to_predict)
            add_features_to_frame('Build seasonality features', df8)

        # Weather
        if self.feature_controller.weather:
            df9 = features.weather.features_weather_sellout(data.df_weather_forecast, data.df_sell_out_stores,
                                                            data.df_cities_master_data, dates_to_predict)

            add_features_to_frame('Build weather features', df9)

        # Store location statistics
        if self.feature_controller.store_location_statistics:
            df10 = features.zone_stats.build_store_location_stats_features(
                dates_when_predicting, data.df_sell_out_stores, data.df_stats_unemployment_city,
                data.df_stats_unemployment_department, data.df_stats_population_city, data.df_stats_income_city)
            add_features_to_frame('Add store location statistics features', df10)

        # Events
        if self.feature_controller.events:
            df11 = features.events.build_sell_out_events_features(dates_to_predict, dates_when_predicting,
                                                                  data.df_sell_out_stores, data.df_events,
                                                                  data.df_cities_master_data,
                                                                  max_distance_km=50, horizon_months=1)
            add_features_to_frame('Add events features', df11)

        consolidated_feature_table = self.consolidate_features(feature_tables)
        Logger.info('Build consolidated features table', 'done', self.__class__.__name__)

        return consolidated_feature_table

    def build_train(self, data: DataLoader, dates_when_predicting: list, dates_to_predict: list):
        return self.build_all_sell_out_features(data, dates_when_predicting, dates_to_predict, is_training=True)

    def build_predict(self, data: DataLoader, dates_when_predicting: list, dates_to_predict: list):
        return self.build_all_sell_out_features(data, dates_when_predicting, dates_to_predict, is_training=False)
