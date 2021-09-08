import os
import pickle
from util.logger import Logger
from configs.configs import Configs
import preprocessor.names as n
import util.input_output as io
from configs.feature_controller import FeatureController
from util.paths import PATH_DATA_FORMATTED, PATH_MODELS
from difflib import get_close_matches


class DataLoader:

    # Common data frames
    #
    df_events = None
    df_weather_forecast = None
    df_predictions_list = None  # DataFrame containing the combinations to predict for each week

    # Data frame attributes - sell-in
    df_cpg_trade_channel = None
    df_historical_forecasts = None
    df_materials_master_data = None
    df_open_orders = None
    df_plants_location = None
    df_promotions_raw = None
    df_fc_acc = None
    df_rebates = None
    df_sales_activities = None
    df_sell_in_fc_acc = None
    df_sell_out_predictions = None
    df_sku_segmentation = None

    # Data frame attributes - sell-out
    group_ids = None  # Matching of consumer units & lead SKUs (groups with minimal dependency)
    df_cities_master_data = None
    df_ean_master_data = None
    df_sell_out = None
    df_sell_out_stores = None
    df_stats_unemployment_department = None
    df_stats_unemployment_city = None
    df_stats_population_city = None
    df_stats_income_city = None

    # Static files - sell-in
    FILE_NAME_CONSOLIDATED_CUSTOMER_MAPPING_DATA = 'customer_information_cleaned.csv'
    FILE_NAME_CPG_SOLD_TO_MAPPING_DATA = 'cpg_sold_to_clean.csv'
    FILE_NAME_MATERIALS_MASTER_DATA = 'skus_clean.csv'
    FILE_NAME_PLANTS_LOCATION_DATA = 'plants_clean.csv'
    FILE_NAME_TRADE_CHANNEL = 'cpg_sales_office.csv'
    FILE_NAME_SELLOUT_SALES = 'sell_out_dts_clean.csv'
    FILE_NAME_SKU_SEG = 'sku_types.csv'

    # Static files - sell-out
    FILE_NAME_EAN_VOLUME_MASTER_DATA = 'ean_master_data_clean.csv'
    FILE_NAME_EAN_MASTER_DATA = 'ean_product_characteristics.csv'
    FILE_NAME_STORES_LOCATION = 'consolidated_stores_list_scoped_clean.csv'
    FILE_NAME_CUSTOMERS_INFO = 'customer_information_cleaned.csv'
    FILE_NAME_CITIES_MASTER_DATA = 'insee_data_cities_cleaned.csv'
    FILE_NAME_EAN_MATERIAL_MATCH_DATA = 'sellout_mapping.csv'
    FILE_NAME_STATS_UNEMPLOYMENT_DEP = 'stats_departments_quarterly_unemployment_rates.csv'
    FILE_NAME_STATS_UNEMPLOYMENT_CITY = 'stats_employment_zones_quarterly_unemployment_rates.csv'
    FILE_NAME_STATS_INCOME_CITY = 'stats_cities_incomes_clean.csv'
    FILE_NAME_STATS_POPULATION_CITY = 'stats_cities_population_clean.csv'

    # Dynamic files - sell-in - need to be frequently updated
    FILE_NAME_EVENTS = 'events_cleaned.csv'
    FILE_NAME_FUTURMASTER = 'futurmaster_clean.csv'
    FILE_NAME_HISTORICAL_FORECASTS = 'hfa_cleaned.csv'
    FILE_NAME_OPEN_ORDERS_DATA = 'hfa_load_weeks.csv'
    FILE_NAME_SALES_ACTIVITIES_DATA = 'activities_clean.csv'
    FILE_NAME_REBATES_DATA = 'rebates_clean.csv'
    FILE_NAME_SELLOUT_SELLING_MAPPING = 'sellout_sellin_mapping.csv'
    FILE_NAME_SELLOUT_PREDICTIONS = 'output_sell_out_extended.csv'
    FILE_NAME_SELLIN_PREDICTIONS_LIST = 'predictions_list.csv'
    FILE_NAME_WEATHER_DATA = 'weather_clean.csv'

    def __init__(self, configs: Configs, feature_controller: FeatureController=None):
        """
        This object deals with all data loading tasks

        Args:
            configs (Configs): specify which country, data version folder, model (sell-in vs sell-out)
            feature_controller (FeatureController): Object controlling which features (hence which data sources) should
            be included in the model.
        """

        if feature_controller is None:
            feature_controller = FeatureController(configs.is_sell_in_model, dict())

        self.feature_controller = feature_controller
        self.is_sell_in_model = configs.is_sell_in_model

        # Master data files & static files (updated on changes, i.e. new SKUs, CPGs, etc.)
        self.path_data_static = os.path.join(PATH_DATA_FORMATTED, configs.country)

        # Dynamic files (updated every week)
        self.path_data_version = os.path.join(PATH_DATA_FORMATTED, configs.country, configs.main_data_version)

        Logger.info('Files loaded from data directory', self.path_data_version, self.__class__.__name__)

    def load_data(self, main_file):

        self.load_predictions_list()
        self.load_material_master_data()
        self.load_futurmaster_promotion_data()

        # Adjust data loading depending on weather features are switched on or off in configs
        if self.feature_controller.events:
            self.load_events_data()

        if self.feature_controller.weather:
            self.load_weather_data()

        if self.is_sell_in_model:
            self.load_open_orders_data()
            self.load_sell_in_data(main_file)
            self.load_historical_forecasts_data()
            self.load_cpg_trade_channel_data()
            self.load_sku_segmentation_data()

            if self.feature_controller.sales_force_activity_kpis:
                self.load_salesforce_activity_data()

            if self.feature_controller.rebates:
                self.load_rebates_data()

            if self.feature_controller.sell_out_inventory or self.feature_controller.sell_out_volumes:
                self.load_sell_out_output_for_sell_in()
        else:
            self.load_ean_master_data()  # Done 1st because we need the conversion to HL in sell-out loader
            self.load_sell_out_data(main_file)
            self.load_stores_master_data()  # Location of all known sell-out stores (incl. official city code)
            self.df_sell_out = self.add_store_ids_to_df_sell_out()
            self.load_stores_location_statistics()

    def load_customer_data(self):
        return io.read_csv(self.path_data_static, DataLoader.FILE_NAME_CONSOLIDATED_CUSTOMER_MAPPING_DATA)

    def load_cpg_trade_channel_data(self):
        self.df_cpg_trade_channel = (io.read_csv(
            self.path_data_static, DataLoader.FILE_NAME_TRADE_CHANNEL, delimiter=';')
                                     .rename(columns={'Customer Planning Group': n.FIELD_CUSTOMER_GROUP,
                                                      'Load Week': n.FIELD_CUSTOMER_GROUP_NAME})
                                     )

        Logger.info('Loaded CPG on/off trade data from file',
                    DataLoader.FILE_NAME_TRADE_CHANNEL, self.__class__.__name__)

    def load_ean_master_data(self):
        relevant_columns = [
            n.FIELD_SKU_EAN,
            # 'Single item ean',
            # 'sku_description',
            'primary_packaging_size_cl',
            'secondary_packaging_size_units',
            'convertion_volume_l',
        ]
        df_ean_volume_convertion = (io.read_csv(self.path_data_static, DataLoader.FILE_NAME_EAN_VOLUME_MASTER_DATA)
                                    .filter(relevant_columns))

        relevant_columns = [
            n.FIELD_SKU_EAN,
            'brand_name',
            'container_type',
            'global_bev_cat_name',
            # 'is_lead_sku',
            'material_sub_group',
            'alcohol_percentage',
            'unit_per_pack',
            # 'pack_size_name'
        ]
        df_ean_master_data = (io.read_csv(self.path_data_static, DataLoader.FILE_NAME_EAN_MASTER_DATA)
                              .filter(relevant_columns))

        for col in ['brand_name', 'material_sub_group', 'global_bev_cat_name']:
            df_ean_master_data[col] = df_ean_master_data[col].str.lower().str.split().str.join(sep='_')

        self.df_ean_master_data = df_ean_volume_convertion.merge(df_ean_master_data, on=n.FIELD_SKU_EAN, how='left')

    def load_events_data(self):
        self.df_events = io.read_csv(self.path_data_version, DataLoader.FILE_NAME_EVENTS)
        Logger.info('Loaded events data from file', DataLoader.FILE_NAME_EVENTS, self.__class__.__name__)

    def load_futurmaster_promotion_data(self):
        self.df_promotions_raw = io.read_csv(self.path_data_version, DataLoader.FILE_NAME_FUTURMASTER)
        Logger.info('Loaded promotions data from file',
                    DataLoader.FILE_NAME_FUTURMASTER, self.__class__.__name__)

    def load_historical_forecasts_data(self):
        self.df_historical_forecasts = io.read_csv(self.path_data_version, DataLoader.FILE_NAME_HISTORICAL_FORECASTS)
        Logger.info('Loaded historical forecasts data from file',
                    DataLoader.FILE_NAME_HISTORICAL_FORECASTS, self.__class__.__name__)

    def load_match_consumer_unit_ean_to_material_ids(self):
        Logger.info('Loading consumer unit EAN to material ids from file',
                    DataLoader.FILE_NAME_EAN_MATERIAL_MATCH_DATA, DataLoader.__name__)

        return (io.read_csv(self.path_data_static, DataLoader.FILE_NAME_EAN_MATERIAL_MATCH_DATA,
                            usecols=['MATNR', 'Sell_Out_EAN'])
                .rename(columns={'MATNR': 'material', 'Sell_Out_EAN': n.FIELD_SKU_EAN}))

    def load_match_lead_sku_to_material_ids(self):
        Logger.info('Loading lead SKU to material ids from file',
                    DataLoader.FILE_NAME_MATERIALS_MASTER_DATA, DataLoader.__name__)

        # All material ids for one lead sku (-1 lead sku?)
        return (io.read_csv(self.path_data_static, DataLoader.FILE_NAME_MATERIALS_MASTER_DATA,
                            usecols=[n.FIELD_SKU_ID, n.FIELD_LEAD_SKU_ID])
                .drop_duplicates())

    def load_material_master_data(self):
        relevant_columns = [
            'material',
            'lead_sku',
            'unit_per_pack',
            'global_bev_cat_name',
            'base_unit_of_measure_characteristic',
            'sku_volume_in_litres',
            'alcohol_percentage',
            'number_of_base_units_per_pallet',
            'brand_name',
            'sub_brand_name'
        ]
        self.df_materials_master_data = \
            io.read_csv(self.path_data_static, DataLoader.FILE_NAME_MATERIALS_MASTER_DATA, usecols=relevant_columns)
        Logger.info('Loaded materials master data from file',
                    DataLoader.FILE_NAME_MATERIALS_MASTER_DATA, self.__class__.__name__)

    def load_predictions_list(self):
        """ Method to load the user-defined list of predictions that should be outputted by our model """
        if os.path.isfile(os.path.join(self.path_data_version, DataLoader.FILE_NAME_SELLIN_PREDICTIONS_LIST)):

            relevant_columns = [
                'date_to_predict',
                n.FIELD_CUSTOMER_GROUP,
                n.FIELD_PLANT_ID,
                n.FIELD_LEAD_SKU_ID
            ]
            self.df_predictions_list = io.read_csv(self.path_data_version, DataLoader.FILE_NAME_SELLIN_PREDICTIONS_LIST,
                                                   usecols=relevant_columns)
            Logger.info('Loaded list of predictions data from file',
                        DataLoader.FILE_NAME_SELLIN_PREDICTIONS_LIST, self.__class__.__name__)

    def load_open_orders_data(self):
        # from hfa_open_orders file
        self.df_open_orders = io.read_csv(self.path_data_version, DataLoader.FILE_NAME_OPEN_ORDERS_DATA)
        Logger.info('Loaded open orders data from file', DataLoader.FILE_NAME_OPEN_ORDERS_DATA, self.__class__.__name__)

    def load_rebates_data(self):
        self.df_rebates = io.read_csv(self.path_data_version, DataLoader.FILE_NAME_REBATES_DATA)

        # Exclude indirect customer rebates
        self.df_rebates = self.df_rebates[~self.df_rebates[n.FIELD_CUSTOMER_GROUP].isin(['F30', 'F31'])]

        Logger.info('Loaded rebates data from file', DataLoader.FILE_NAME_REBATES_DATA, self.__class__.__name__)

    def load_salesforce_activity_data(self):
        relevant_columns = [
            n.FIELD_YEARMONTH,
            'sold_to_party_id',
            'category_name',
            # 'status_name',
            'no_of_activities'
        ]
        self.df_sales_activities = io.read_csv(self.path_data_version, DataLoader.FILE_NAME_SALES_ACTIVITIES_DATA,
                                               usecols=relevant_columns)
        self.df_sales_activities = self.df_sales_activities[self.df_sales_activities['sold_to_party_id'].notnull()]

        sold_to_mapping = (io.read_csv(self.path_data_static, DataLoader.FILE_NAME_CPG_SOLD_TO_MAPPING_DATA,
                                       usecols=[n.FIELD_CUSTOMER_GROUP, 'sold_to_party_id'])
                           .set_index('sold_to_party_id'))

        relevant_columns = [
            n.FIELD_YEARMONTH,
            n.FIELD_CUSTOMER_GROUP,
            'category_name',
            # 'status_name',
            'no_of_activities'
        ]

        self.df_sales_activities = self.df_sales_activities.merge(sold_to_mapping, on='sold_to_party_id', how='inner')
        self.df_sales_activities = (self.df_sales_activities[self.df_sales_activities[n.FIELD_CUSTOMER_GROUP].notnull()]
                                    .filter(relevant_columns))

        Logger.info('Loaded SalesForce activities data from file',
                    DataLoader.FILE_NAME_SALES_ACTIVITIES_DATA, self.__class__.__name__)

    def load_sell_in_data(self, *path):
        self.df_sell_in_fc_acc = (io.read_csv(self.path_data_version, *path)
                                  .rename(columns={'total_shipments': 'total_shipments_volume'}))
        Logger.info('Loaded sell-in data from file', *path, self.__class__.__name__)

    def load_sell_in_forecast_accuracy_comparison_data(self, file_name):
        self.df_fc_acc = io.read_csv(self.path_data_version, file_name)
        Logger.info('Loaded historical forecast accuracies from file', file_name, self.__class__.__name__)

    def load_sell_out_data(self, *path):

        class_name = self.__class__.__name__

        relevant_columns = [
            n.FIELD_CUSTOMER,
            n.FIELD_SKU_EAN,
            n.FIELD_STORE_NAME,
            n.FIELD_YEARMONTH,
            n.FIELD_QTY_SOLD_SU,
            # 'sku_description',
            # n.FIELD_CITY,
            # 'zip_code'
        ]

        data = (io.read_csv(self.path_data_version, *path)
                .rename(columns={'year_month': n.FIELD_YEARMONTH})
                .filter(relevant_columns)
                )
        data[n.FIELD_CUSTOMER] = data[n.FIELD_CUSTOMER].str.upper()
        data = data[~data[n.FIELD_CUSTOMER].isin(['LEADERPRICE', 'FRANPRIX'])]
        Logger.warning('Dropping Leaderprice & Intermarche sell-out - No data available for 2018', class_name=class_name)

        missing_ean = set(data[n.FIELD_SKU_EAN]).difference(self.df_ean_master_data[n.FIELD_SKU_EAN])
        nb_rows = round(data[data[n.FIELD_SKU_EAN].isin(missing_ean)].shape[0] / data.shape[0] * 100, 2)
        Logger.warning('{}% of rows dropped () | {} unknow eans in'.format(nb_rows, len(missing_ean)), *path, class_name)

        data = data[~data[n.FIELD_SKU_EAN].isin(missing_ean)]
        del nb_rows, missing_ean

        col_volume = 'convertion_volume_l'
        conversion = (data[[n.FIELD_SKU_EAN]]
                      .merge(self.df_ean_master_data.drop_duplicates([n.FIELD_SKU_EAN, col_volume]),
                             on=n.FIELD_SKU_EAN, how='left')
                      )[col_volume].values

        data[n.FIELD_LABEL] = data[n.FIELD_QTY_SOLD_SU] * conversion / 100  # Volume in hectoliters
        Logger.info('Loaded sell-out data from file', *path, class_name)

        relevant_columns = [
            n.FIELD_SKU_EAN,
            n.FIELD_CUSTOMER,
            n.FIELD_STORE_NAME,
            n.FIELD_YEARMONTH,
            n.FIELD_LABEL
        ]
        self.df_sell_out = data[relevant_columns]

    def load_sell_out_output_for_sell_in(self):
        self.df_sell_out_predictions = io.read_csv(self.path_data_version, DataLoader.FILE_NAME_SELLOUT_PREDICTIONS)
        Logger.info('Loaded latest sell-out predictions from file', DataLoader.FILE_NAME_SELLOUT_PREDICTIONS,
                    self.__class__.__name__)

        self.load_ean_master_data()  # Done 1st because we need the conversion to HL in sell-out loader
        self.load_sell_out_data(DataLoader.FILE_NAME_SELLOUT_SALES)
        self.df_sell_out = self.add_cpg_to_sell_out_data(self.df_sell_out)

        # Minimal independent groups of EANs and Lead SKUs (after matching)
        with open(os.path.join(PATH_MODELS, 'grouping_ean_skus.pkl'), 'rb') as f:
            self.group_ids = pickle.load(f)

    def load_sku_segmentation_data(self):
        self.df_sku_segmentation = io.read_csv(self.path_data_static, DataLoader.FILE_NAME_SKU_SEG)
        Logger.info('Loaded sku segmentation data from file', DataLoader.FILE_NAME_SKU_SEG, self.__class__.__name__)

    def load_stores_location_statistics(self):
        class_name = self.__class__.__name__

        self.df_stats_unemployment_department = \
            io.read_csv(self.path_data_static, DataLoader.FILE_NAME_STATS_UNEMPLOYMENT_DEP)
        Logger.info('Loaded Insee statistics (1/4) from file', DataLoader.FILE_NAME_STATS_UNEMPLOYMENT_DEP, class_name)

        self.df_stats_unemployment_city = io.read_csv(self.path_data_static, DataLoader.FILE_NAME_STATS_UNEMPLOYMENT_CITY)
        Logger.info('Loaded Insee statistics (2/4) from file', DataLoader.FILE_NAME_STATS_UNEMPLOYMENT_CITY, class_name)

        self.df_stats_population_city = io.read_csv(self.path_data_static, DataLoader.FILE_NAME_STATS_POPULATION_CITY)
        Logger.info('Loaded Insee statistics (3/4) from file', DataLoader.FILE_NAME_STATS_POPULATION_CITY, class_name)

        self.df_stats_income_city = io.read_csv(self.path_data_static, DataLoader.FILE_NAME_STATS_INCOME_CITY)
        Logger.info('Loaded Insee statistics (4/4) from file', DataLoader.FILE_NAME_STATS_INCOME_CITY, class_name)

    def load_stores_master_data(self):
        relevant_columns = [
            n.FIELD_CUSTOMER,
            n.FIELD_STORE_NAME,
            n.FIELD_STORE_ID,
            n.FIELD_CITY,
            n.FIELD_ZIP_CODE,
            n.FIELD_CITY_CODE,
        ]
        self.df_sell_out_stores = io.read_csv(self.path_data_static, DataLoader.FILE_NAME_STORES_LOCATION,
                                              usecols=relevant_columns)
        Logger.info('Loaded stores location master data from file', DataLoader.FILE_NAME_STORES_LOCATION,
                    self.__class__.__name__)

    def load_weather_data(self):
        self.df_weather_forecast = (io.read_csv(self.path_data_version, DataLoader.FILE_NAME_WEATHER_DATA)
                                    .rename(columns={'lat': n.FIELD_LATITUDE, 'lon': n.FIELD_LONGITUDE}))
        Logger.info('Loaded weather forecast data from file',
                    DataLoader.FILE_NAME_WEATHER_DATA, self.__class__.__name__)

        if self.is_sell_in_model:
            self.df_plants_location = (io.read_csv(self.path_data_static, DataLoader.FILE_NAME_PLANTS_LOCATION_DATA)
                                       .rename(columns={'lat': n.FIELD_LATITUDE, 'lon': n.FIELD_LONGITUDE}))
            Logger.info('Loaded plant location data from file',
                        DataLoader.FILE_NAME_PLANTS_LOCATION_DATA, self.__class__.__name__)
        else:
            self.df_cities_master_data = io.read_csv(self.path_data_static, DataLoader.FILE_NAME_CITIES_MASTER_DATA)
            Logger.info('Loaded cities master data from file',
                        DataLoader.FILE_NAME_CITIES_MASTER_DATA, self.__class__.__name__)

    def add_cpg_to_sell_out_data(self, sell_out):

        class_name = self.__class__.__name__

        # Add CPG to sell-out data
        match = dict()
        cpg_match = (self.load_customer_data()[[n.FIELD_BRAND, n.FIELD_CUSTOMER_GROUP]].drop_duplicates()
                     .set_index(n.FIELD_BRAND)
                     .T.to_dict('list'))

        for cpg in sell_out[n.FIELD_CUSTOMER].unique():
            closest_cpg = get_close_matches(cpg, cpg_match.keys())[0]
            # print(' <> '.join([cpg, closest_cpg]))
            match.update({cpg: cpg_match[closest_cpg][0]})

        sell_out[n.FIELD_CUSTOMER_GROUP] = sell_out[n.FIELD_CUSTOMER].map(match)
        Logger.info('Added CPG info to sell-out data from file', DataLoader.FILE_NAME_CUSTOMERS_INFO, class_name)

        return sell_out

    def add_store_ids_to_df_sell_out(self):

        stores_columns = [n.FIELD_CUSTOMER, n.FIELD_STORE_NAME, n.FIELD_STORE_ID, n.FIELD_CITY_CODE]
        data = (self.df_sell_out
                .drop(columns=[n.FIELD_CITY, n.FIELD_ZIP_CODE], errors='ignore')
                .merge(self.df_sell_out_stores[stores_columns],
                       on=[n.FIELD_CUSTOMER, n.FIELD_STORE_NAME], how='left')
                )

        missing_stores = data[data[n.FIELD_CITY_CODE].isnull()]
        if missing_stores.shape[0]:
            Logger.info('Number of stores ignored (no match with store location master data)',
                        str(missing_stores.drop_duplicates([n.FIELD_CUSTOMER, n.FIELD_STORE_NAME]).shape[0]),
                        self.__class__.__name__)

            data = data[data[n.FIELD_CITY_CODE].notnull()]

        return data[[n.FIELD_SKU_EAN, n.FIELD_CUSTOMER, n.FIELD_STORE_ID, n.FIELD_YEARMONTH, n.FIELD_LABEL]]

    def generate_default_predictions_list(self):

        if not self.is_sell_in_model:
            relevant_columns = [
                n.FIELD_SKU_EAN,
                n.FIELD_CUSTOMER,
                n.FIELD_STORE_ID,
                n.FIELD_YEARMONTH
            ]
            return (self.df_sell_out.filter(relevant_columns).drop_duplicates()
                    .rename(columns={n.FIELD_YEARMONTH: 'date_to_predict'}))

        relevant_columns = [
            n.FIELD_LEAD_SKU_ID,
            n.FIELD_CUSTOMER_GROUP,
            n.FIELD_PLANT_ID,
            n.FIELD_YEARWEEK
        ]
        return (self.df_sell_in_fc_acc.filter(relevant_columns).drop_duplicates()
                .rename(columns={n.FIELD_YEARWEEK: 'date_to_predict'}))
