import logging
import pandas as pd
from functools import reduce
import preprocessor.names as n
from util.misc import create_list_period, diff_period, get_nbsales_zeros


def features_amount_sales(data, dates_when_predicting, dates_to_predict, timelag=52, is_weekly: bool=True):
    """
    """
    to_merge = pd.DataFrame()
    for i in range(len(dates_when_predicting)):
        filter_col = [col for col in data if (col <= dates_when_predicting[i])]
        datatemp = data[filter_col]
        datatemp = datatemp[datatemp.columns[-timelag:]]
        datatemp.columns = map('sales{}'.format, range(timelag, 0, -1))
        datatemp['nbperiod_nosales'] = get_nbsales_zeros(datatemp)
        datatemp['forward_steps'] = diff_period(dates_when_predicting[i], dates_to_predict[i], is_weekly)
        datatemp['date_when_predicting'] = dates_when_predicting[i]

        datatemp['date_to_predict'] = dates_to_predict[i]
        datatemp.set_index('date_when_predicting', append=True, inplace=True)
        datatemp.set_index('date_to_predict', append=True, inplace=True)
        to_merge = pd.concat([to_merge, datatemp])

    return to_merge.reset_index()


def features_labels(data, dates_when_predicting, dates_to_predict):
    """
    Adding the groundtruth column to the predicitons
    :param data: dataframe containing sales data
    :param dates_when_predicting: the dates at which the predicitons needs to be made
    :param dates_to_predict: the dates for which we are trying to predict the amount of shipments
    :return:
    """
    to_merge = pd.DataFrame()
    for i in range(len(dates_to_predict)):
        datatemp = data[dates_to_predict[i]].to_frame()
        datatemp.columns = [n.FIELD_LABEL]
        datatemp['date_when_predicting'] = dates_when_predicting[i]
        datatemp.set_index('date_when_predicting', append=True, inplace=True)

        datatemp['date_to_predict'] = dates_to_predict[i]
        datatemp.set_index('date_to_predict', append=True, inplace=True)
        to_merge = pd.concat([to_merge, datatemp])

    return to_merge.reset_index()


def create_rolling_shipments(hist_forecast, rolling_granularity, time_granularity, column_name):
    assert time_granularity in [n.FIELD_YEARMONTH, n.FIELD_YEARWEEK]

    column_feature = 'total_shipments'
    gb_cols = rolling_granularity + [time_granularity]
    total_sh = hist_forecast.groupby(gb_cols, as_index=False).agg({column_feature: 'sum'})

    if time_granularity == n.FIELD_YEARWEEK:
        rolling_wdw = 52
    else:
        rolling_wdw = 12

    total_sh[column_name] = (total_sh.groupby(rolling_granularity, as_index=False)[column_feature]
                             .rolling(rolling_wdw)
                             .sum()
                             .values)
    total_sh = (total_sh
                .rename(columns={time_granularity: 'date_when_predicting'})
                .drop(columns=[column_feature]))

    return total_sh


def create_rolling_sales_volume(data_sell_out, rolling_granularity, time_granularity, column_name):
    assert time_granularity in [n.FIELD_YEARMONTH, n.FIELD_YEARWEEK]

    column_feature = n.FIELD_LABEL
    gb_cols = rolling_granularity + [time_granularity]
    total_sales = data_sell_out.groupby(gb_cols, as_index=False).agg({column_feature: 'sum'})

    if time_granularity == n.FIELD_YEARWEEK:
        rolling_wdw = 52
    else:
        rolling_wdw = 12

    total_sales[column_name] = (total_sales
                                .groupby(rolling_granularity, as_index=False)[column_feature]
                                .rolling(rolling_wdw)
                                .sum()
                                .values)
    total_sales = (total_sales
                   .rename(columns={time_granularity: 'date_when_predicting'})
                   .drop(columns=[column_feature]))

    return total_sales


def create_rolling_sales_volume_per_category(data_actual_sales, data_sku_master_data,
                                             rolling_granularity, time_granularity, column_name):
    """
    Method to compute features per brand or sub-brand (for sell-in & sell-out).

    Args:
        data_actual_sales (pd.DataFrame): Actual sell-in or sell-out volumes sold
        data_sku_master_data (pd.DataFrame): table containing match between lead_sku / ean and brand / sub-brand
    """
    assert time_granularity in [n.FIELD_YEARMONTH, n.FIELD_YEARWEEK]

    column_sku_id, column_feature = n.FIELD_SKU_EAN, n.FIELD_LABEL

    columns_sku = set(rolling_granularity).intersection({n.FIELD_SKU_BRAND, n.FIELD_SKU_SUB_BRAND})
    columns_sku.add(column_sku_id)
    data_sku_master_data = data_sku_master_data.filter(columns_sku)

    data_actual_sales = (data_actual_sales
                         .merge(data_sku_master_data, on=n.FIELD_SKU_EAN, how='left'))

    column_feature = n.FIELD_LABEL
    gb_cols = rolling_granularity + [time_granularity]
    total_sales = data_actual_sales.groupby(gb_cols, as_index=False).agg({column_feature: 'sum'})

    if time_granularity == n.FIELD_YEARWEEK:
        rolling_wdw = 52
    else:
        rolling_wdw = 12

    total_sales[column_name] = (total_sales
                                .groupby(rolling_granularity, as_index=False)[column_feature]
                                .rolling(rolling_wdw)
                                .sum()
                                .values)
    total_sales = (total_sales
                   .rename(columns={time_granularity: 'date_when_predicting'})
                   .drop(columns=[column_feature]))

    total_sales = (total_sales
                   .merge(data_actual_sales.filter(columns_sku).drop_duplicates(column_sku_id),
                          on=[n.FIELD_SKU_BRAND], how='outer')
                   .drop(columns=[n.FIELD_SKU_BRAND])
                   )

    return total_sales
