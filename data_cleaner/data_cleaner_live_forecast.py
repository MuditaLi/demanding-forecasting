import os
import pandas as pd
import numpy as np
import preprocessor.names as n
import util.input_output as io
from util.paths import PATH_DATA_RAW, PATH_DATA_FORMATTED
from data_cleaner.data_cleaner import clean_col_names, clean_lead_sku


def clean_hfa_load_weeks_report_20190429(country, folder_new_extract, folder_latest_extract,
                                         date_when_predicting: int, sep_thousand=','):
    files = ['FR_WEEK_1_52_2019_29_4_2019.csv']
    results = list()

    for file in files:
        data = pd.read_csv(os.path.join(
            PATH_DATA_RAW, 'HistoricalForecast2015_2019', 'FR_WEEK_1_52_2019_29_4_2019.csv'), sep=';', dtype=str)

        data = (clean_col_names(data)
                .rename(columns={'calendar_year__week': n.FIELD_YEARWEEK})
                )
        data[n.FIELD_LEAD_SKU_ID] = clean_lead_sku(data[n.FIELD_LEAD_SKU_ID])
        results.append(data)

    data = pd.concat(results,  axis=0)

    numerical_columns = [
        'total_shipments',
        'open_orders',
        'promotion_uplift',
        'apo_calculated_sales_estimate',
        'total_sales_volume'
    ]
    for col in numerical_columns:
        data[col] = data[col].str.replace(sep_thousand, '').astype(float)

    integer_columns = [
        'load_week',
        n.FIELD_YEARWEEK,
        n.FIELD_YEARMONTH
    ]
    for col in integer_columns:
        data[col] = data[col].apply(lambda x: ''.join(x.split('.')[::-1])).astype(int)

    data = data[
        [n.FIELD_YEARMONTH,
         n.FIELD_YEARWEEK,
         'load_week',
         n.FIELD_CUSTOMER_GROUP,
         n.FIELD_LEAD_SKU_ID,
         n.FIELD_PLANT_ID,
         'total_shipments',
         'open_orders',
         'promotion_uplift',
         'total_sales_volume',
         'apo_calculated_sales_estimate'
         ]
    ]

    # Create table containing list of all combinations to predict
    tmp = (data[data['load_week'] <= data[n.FIELD_YEARWEEK]]
           .filter([n.FIELD_YEARWEEK, n.FIELD_CUSTOMER_GROUP, n.FIELD_PLANT_ID, n.FIELD_LEAD_SKU_ID])
           .drop_duplicates()
           .rename(columns={n.FIELD_YEARWEEK: 'date_to_predict'})
           )
    io.write_csv(tmp, PATH_DATA_FORMATTED, country, folder_new_extract, 'predictions_list.csv')

    # hfa_load_weeks (# TODO - SOME LOAD WEEKS ARE MISSING - FOR LIVE FORECAST WE ONLY CARE ABOUT LATEST LOAD WEEK)
    # TODO - CAREFUL WE NEED TO FORCE THE LOAD WEEK TO BE EQUAL TO DATE WHEN PREDICTING
    data['load_week'] = data['load_week'].where(data['load_week'] <= date_when_predicting, data['load_week'] - 1)
    data = data.drop(columns=['apo_calculated_sales_estimate'])
    io.write_csv(data, PATH_DATA_FORMATTED, country, folder_new_extract, 'hfa_load_weeks.csv')

    # Append to hfa_cleaned (table containing all actual sales until now)
    # data = data[data['load_week'] > data[n.FIELD_YEARWEEK]]
    data_previous_extract = io.read_csv(PATH_DATA_FORMATTED, country, folder_latest_extract, 'hfa_cleaned.csv')

    data_previous_extract = data_previous_extract[
        (data_previous_extract[n.FIELD_YEARWEEK] < data_previous_extract['load_week'])
        &
        (data_previous_extract[n.FIELD_YEARWEEK] < data[n.FIELD_YEARWEEK].min())
    ]

    # Check consistency between extracts
    # tmp1 = data_previous_extract[
    #     (data_previous_extract[n.FIELD_YEARWEEK] >= 201901)
    #     &
    #     (data_previous_extract[n.FIELD_YEARWEEK] < 201907)
    # ]
    # tmp1.groupby(n.FIELD_YEARWEEK)['total_shipments'].sum()
    #
    # tmp2 = data[data[n.FIELD_YEARWEEK].isin(tmp1[n.FIELD_YEARWEEK])]
    # tmp2.groupby(n.FIELD_YEARWEEK)['total_shipments'].sum()
    #
    # tmp = tmp1.merge(tmp2, on=[n.FIELD_YEARMONTH, n.FIELD_YEARWEEK,
    #                            n.FIELD_PLANT_ID, n.FIELD_LEAD_SKU_ID, n.FIELD_CUSTOMER_GROUP], how='outer')
    # tmp = tmp[
    #     (tmp['total_shipments_y'].round(0) != tmp['total_shipments_x'].round(0)) & (tmp['total_shipments_x'] != 0)]

    data = pd.concat([data, data_previous_extract], axis=0, sort=False)
    io.write_csv(data, PATH_DATA_FORMATTED, country, folder_new_extract, 'hfa_cleaned.csv')


if __name__ == '__main__':

    country, folder_new_extract, folder_latest_extract, date_when_predicting, sep_thousand = \
        'france', '201918', '201901', 201917, '.'

    clean_hfa_load_weeks_report_20190429(
        country, folder_new_extract, folder_latest_extract, date_when_predicting, sep_thousand)
