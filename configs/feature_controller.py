class SellInController:

    cpg_trade_channel_indicator = True
    events = False
    open_orders = True
    open_orders_sum_to_date_to_predict = True
    plant_cpg_indicators = True
    promotion_uplift = True
    promotion_plan = True  # raw promotion features
    promotion_cannibalization = True
    rebates = False
    sales_customer_last_year_rolling = True
    sales_customer_sku_last_year_rolling = True
    sales_customer_sku_plant_last_year_rolling = True
    sales_force_activity_kpis = False
    seasonality = True
    sell_out_volumes = False
    sell_out_inventory = False
    shipped_orders = True
    shipped_orders_sum_to_date_to_predict = False
    sku_characteristics = True
    sku_segmentation = True  # Festival or promotion specific SKUs, or NPDs
    weather = True

    def __init__(self, features_controller: dict):
        """
        Which features should be included in the sell-in model. Default values are based on assessment of
        model performance during the French POC project.

        Args:
            features_controller (dict): Updated settings {name_attribute: True (Feature ON) or False (Feature OFF)}
        """
        for name, value in features_controller.items():
            if name in vars(SellInController):
                self.__setattr__(name, bool(value))


class SellOutController:

    consumer_unit_characteristics = True
    events = True
    sales_customer_last_year_rolling = True
    sales_customer_brand_last_year_rolling = True
    sales_customer_consumer_unit_last_year_rolling = True
    sales_customer_store_last_year_rolling = True
    sales_customer_store_consumer_unit_last_year_rolling = True
    seasonality = True
    store_location_statistics = True
    weather = True

    def __init__(self, features_controller: dict):
        """
        Which features should be included in the sell-in model. Default values are based on assessment of
        model performance during the French POC project.

        Args:
            features_controller (dict): Updated settings {name_attribute: True (Feature ON) or False (Feature OFF)}
        """

        for name, value in features_controller.items():
            if name in vars(SellOutController):
                self.__setattr__(name, bool(value))


class FeatureController(SellInController or SellOutController):
    """
    Object to control which features should be switched on or off to train a model and predict
    Also used to switch loading of certain data objects off whenever the corresponding features are switched off.
    """

    def __init__(self, is_sell_in_model: bool, feature_controller: dict):
        """
        Args:
            is_sell_in_model (bool): Model to consider | True = sell-in vs False = sell-out
            feature_controller (dict): dictionary containing all features to switch on
        """
        features_controller = (SellInController if is_sell_in_model else SellOutController)(feature_controller)
        for name, value in features_controller.__dict__.items():
            self.__setattr__(name, bool(value))

