import pandas as pd
import features
import util.misc as misc
import preprocessor.names as n
from dcd_forecast_model.feature_builder import FeatureEngineering
from preprocessor.data_loader import DataLoader
from util.logger import Logger


class FeatureEngineeringSellIn(FeatureEngineering):
    """
    This class loads the data, build features and stack them to build the training or predict set
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_all_sell_in_features(self,
                                   data: DataLoader,
                                   dates_when_predicting: list,
                                   dates_to_predict: list,
                                   is_training: bool) -> pd.DataFrame:

        feature_tables = list()
        df_materials = misc.convert_categorical(data.df_materials_master_data.drop(columns=['lead_sku']))
        df_sell_in_data = data.df_sell_in_fc_acc.groupby(self.granularity)['total_shipments_volume'].sum().unstack(
            n.FIELD_YEARWEEK).fillna(0)

        def add_features_to_frame(message, df: [pd.DataFrame], name=self.__class__.__name__):
            """ helper function to append latest features data frame & print custom log"""
            if not df.empty:
                feature_tables.append(df)
            Logger.info(message, 'done', name)

        # Reference data frame giving all relevant plant / cpg / sku combinations. Always first table to build (or df1)
        if data.df_predictions_list is not None:
            df0 = features.predictions.filter_predictions_list(data.df_predictions_list, dates_to_predict)
            add_features_to_frame('Predictions list used as reference point', df0)

        if self.feature_controller.sales_customer_sku_plant_last_year_rolling:
            df1 = features.sales.features_amount_sales(
                df_sell_in_data, dates_when_predicting, dates_to_predict,
                timelag=52 if self.is_weekly_forecast else 12,
                is_weekly=self.is_weekly_forecast
            )
            add_features_to_frame('Build sales features', df1)

        # Actual volumes sold (dependant variable)
        # Only build for training or back-testing of our model
        if df_sell_in_data.columns.max() >= max(dates_to_predict):
            df2 = features.sales.features_labels(df_sell_in_data, dates_when_predicting, dates_to_predict)
            add_features_to_frame('Add target volume', df2)

        # SKU characteristics
        if self.feature_controller.sku_characteristics:
            # During training, pipelines (default missing values) are created and stored for inference phase.
            # Most frequent categories | Median values for numerical quantities
            if is_training:
                df3, materials_pipeline = features.materials.features_materials(df_materials, is_training)
                self.pipelines['materials'] = materials_pipeline
            else:
                df3, _ = features.materials.features_materials(df_materials, is_training, self.pipelines['materials'])
            add_features_to_frame('Build SKU describing features', df3)

        # Plants & CPG features
        if self.feature_controller.plant_cpg_indicators:
            if is_training:
                df8, plant_pipeline = features.plant_cpg.features_plant(data.df_sell_in_fc_acc, is_training)
                self.pipelines['plant'] = plant_pipeline
            else:
                df8, _ = features.plant_cpg.features_plant(data.df_sell_in_fc_acc, is_training, self.pipelines['plant'])
            add_features_to_frame('Build plant and CPG category features', df8)

        # Business activity features
        if self.feature_controller.sales_customer_last_year_rolling:
            df4 = features.sales.create_rolling_shipments(data.df_historical_forecasts,
                                                          [n.FIELD_CUSTOMER_GROUP], n.FIELD_YEARWEEK,
                                                          'cust_total_shipments_LY_rw')
            add_features_to_frame('Build business activity features (1/2)', df4)

        if self.feature_controller.sales_customer_sku_last_year_rolling:
            df5 = features.sales.create_rolling_shipments(data.df_historical_forecasts,
                                                          [n.FIELD_CUSTOMER_GROUP, n.FIELD_LEAD_SKU_ID],
                                                          n.FIELD_YEARWEEK, 'cust_sku_total_shipments_LY_rw')
            add_features_to_frame('Build business activity features (2/2)', df5)

        # Promotion related features
        if self.feature_controller.promotion_uplift:
            df6 = features.promo.features_promos(data.df_historical_forecasts, self.granularity, dates_to_predict)
            add_features_to_frame('Build promotion related features', df6)

        # Promo raw
        if self.feature_controller.promotion_plan:
            df14 = features.promo.features_promos_raw(data.df_promotions_raw, dates_to_predict, dist=4)
            add_features_to_frame('Build raw promo feature (1/2)', df14)

        if self.feature_controller.promotion_cannibalization:
            df15 = features.promo.features_promos_cannib(data.df_sell_in_fc_acc, data.df_materials_master_data,
                                                         data.df_promotions_raw, dates_to_predict, dist=3)
            add_features_to_frame('Build raw promo feature (2/2)', df15)

        # Open orders features
        if self.feature_controller.open_orders:
            df7 = features.open_orders.features_open_orders(
                data.df_open_orders, self.granularity, dates_when_predicting, dates_to_predict)
            add_features_to_frame('Build open orders related features', df7)

        if self.feature_controller.open_orders_sum_to_date_to_predict:
            df16 = features.open_orders.feature_sumopen_orders(
                data.df_open_orders, self.granularity, dates_when_predicting, dates_to_predict)
            add_features_to_frame('Build sum open orders feature', df16)

        # Known orders features
        if self.feature_controller.shipped_orders:
            df10 = features.known_orders.features_known_orders(
                data.df_open_orders, self.granularity, dates_when_predicting, dates_to_predict)
            # df10['sumtotal_orders'] = df10[n.FIELD_KNOWN_ORDERS] + df7[n.FIELD_OPEN_ORDERS]
            add_features_to_frame('Build actual shipments related features', df10)

        if self.feature_controller.shipped_orders_sum_to_date_to_predict:
            df17 = features.known_orders.feature_sumknown_orders(data.df_open_orders, self.granularity,
                                                                 dates_when_predicting, dates_to_predict)
            add_features_to_frame('Build sum known orders feature', df17)

        # Seasonality features
        if self.feature_controller.sales_customer_sku_last_year_rolling:
            df9 = features.features_date.create_dates(dates_to_predict)
            add_features_to_frame('Build seasonality features', df9)

        # Weather features
        if self.feature_controller.weather:
            df11 = features.weather.features_weather_sell_in(
                data.df_weather_forecast, data.df_plants_location, dates_to_predict)

            add_features_to_frame('Build weather features', df11)

        # Trade channel
        if self.feature_controller.cpg_trade_channel_indicator:
            df12 = features.trade_channel.build_on_trade_feature(data.df_sell_in_fc_acc, data.df_cpg_trade_channel)
            add_features_to_frame('Build trade channel feature', df12)

        # Sales force activity KPIs
        if self.feature_controller.sales_force_activity_kpis:
            df13 = features.sales_activities.build_salesforce_activity_features(
                data.df_sales_activities, data.df_sell_in_fc_acc, dates_when_predicting)
            add_features_to_frame('Build sales force activity KPIs features', df13)

        # Events features
        if self.feature_controller.events:
            df19 = features.events.build_sell_in_events_features(data.df_events, data.df_plants_location,
                                                                 dates_to_predict, dates_when_predicting)
            add_features_to_frame('Build events features', df19)

        # SKU segmentation features
        if self.feature_controller.sku_segmentation:
            df20 = features.sku_segmentation.build_seg_features(data.df_sell_in_fc_acc, data.df_sku_segmentation)
            add_features_to_frame('Build SKU segmentation features', df20)

        # Rebates features
        if self.feature_controller.rebates:
            df21 = features.rebates.build_rebates_features(dates_when_predicting, dates_to_predict,
                                                           data.df_sell_in_fc_acc, data.df_rebates)
            add_features_to_frame('Build rebates features', df21)

        # Sell-out sales
        if self.feature_controller.sell_out_volumes:
            df18 = features.incorporate_sell_out.incorporate_sell_out(data.df_sell_out_predictions, dates_to_predict)
            add_features_to_frame('Build sell out prediction feature', df18)

        # Sell-out inventory
        if self.feature_controller.sell_out_inventory:
            df22 = features.sell_out_inventory.build_sell_out_inventory_features(
                data.group_ids, data.df_sell_in_fc_acc, data.df_sell_out)
            add_features_to_frame('Build sell-out inventory features', df22)

        consolidated_feature_table = self.consolidate_features(feature_tables)
        Logger.info('Build consolidated features table', 'done', self.__class__.__name__)

        return consolidated_feature_table

    def build_train(self, data: DataLoader, dates_when_predicting: list, dates_to_predict: list):
        return self.build_all_sell_in_features(data, dates_when_predicting, dates_to_predict, is_training=True)

    def build_predict(self, data: DataLoader, dates_when_predicting: list, dates_to_predict: list):
        return self.build_all_sell_in_features(data, dates_when_predicting, dates_to_predict, is_training=False)

    def replace_weather_features_in_feature_table(self, df_test: pd.DataFrame, data: DataLoader, delta_celsius: float=None,
                                                  use_existing_year: int=None):
        """
        Adapt features table to run weather scenarios. Either add constant delta of temperature throughout the period.
        Ideally, please use it with models that only include temperature as weather features.

        Args:
            df_test (pd.DataFrame): test set including all relevant features
            data (DataLoader): all relevant data objects
            delta_celsius (float): list of temperature deltas to apply (in celsius)
            use_existing_year (int): use specified year's weather to predict.
            If we do so, we assume that all weeks in scope belong to the same year.

            Please use either delta_celsius or use_existing_year, but not both.

        Returns:
            df_test (pd.DataFrame): data frame of features
        """
        # Remark: all columns with "_org" settings contains the true data (feature without scenario)
        cols = list(filter(
            lambda x: x.startswith('apparent_temperature_mean_') and not x.endswith('_org'), df_test.columns))

        # Copy original temperature features and true dates to predict
        if not set(df_test.columns).intersection(list(map(lambda x: x + '_org', cols))):
            df_test[list(map(lambda x: x + '_org', cols))] = df_test[cols].copy()

        # Scenario with fixed delta of temperature
        if delta_celsius is not None:
            df_test[cols] = df_test[list(map(lambda x: x + '_org', cols))] + delta_celsius

        # Scenarios with temperature of other years
        elif use_existing_year is not None:

            if 'date_to_predict_org' not in df_test.columns:
                df_test['date_to_predict_org'] = df_test['date_to_predict'].copy()

            df_test = df_test.drop(columns=cols)

            # Build weather features
            true_year = df_test.loc[0, 'date_to_predict_org'] // 100
            fake_dates_to_predict = list(map(lambda x: x - 100 * (true_year - use_existing_year),
                                             df_test['date_to_predict_org'].unique()))

            df = features.weather.features_weather_sell_in(
                data.df_weather_forecast, data.df_plants_location, fake_dates_to_predict)
            df['date_to_predict'] = df['date_to_predict'] + (true_year - use_existing_year) * 100

            df_test = self.consolidate_features([df_test, df]).drop(columns=['date_to_predict_org'])

        return df_test
