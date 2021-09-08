""" Build all store location statistics features """
import math
import itertools
import pandas as pd
from functools import partial
import util.misc as misc
import preprocessor.names as n


def convert_month_to_quarter(yearmonth: int):
    """ Method to convert calendar year month attribute to quarter format """
    return (yearmonth // 100) * 100 + math.ceil((yearmonth % 100) / 3)


def get_all_store_dates_combinations(dates_when_predicting: list, stores: pd.DataFrame,
                                                coordinates: bool=True, dates_to_predict: list=None):
    relevant_columns = [n.FIELD_STORE_ID, n.FIELD_CITY_CODE]
    if coordinates:
        relevant_columns = [n.FIELD_STORE_ID, n.FIELD_LATITUDE, n.FIELD_LONGITUDE]

    stores = stores[relevant_columns]
    features = pd.DataFrame(
        list(itertools.product(stores[n.FIELD_STORE_ID].unique(), dates_when_predicting)),
        columns=[n.FIELD_STORE_ID, 'date_when_predicting']
    )

    if dates_to_predict is not None:
        dates = pd.DataFrame(list(zip(dates_when_predicting, dates_to_predict)),
                             columns=['date_when_predicting', 'date_to_predict'])
        features = features.merge(dates, on='date_when_predicting', how='left')

    # Add iris_code_commune of store
    return features.merge(stores, on=n.FIELD_STORE_ID, how='left')


def add_latest_available_quarterly_statistics(features: pd.DataFrame, stats: pd.DataFrame, col_merge: list, suf='dep'):
    stats = stats[col_merge + ['calendar_yearquarter', 'unemployment_rate']]
    stats.columns = col_merge + ['calendar_yearquarter', '_'.join(['unemployment_rate', suf])]
    tmp = features.merge(stats, on=col_merge, how='left', suffixes=['', '_stats'])
    tmp.sort_values(['quarter', 'calendar_yearquarter'], ascending=False, inplace=True)
    tmp = tmp[tmp['quarter'] > tmp['calendar_yearquarter']].drop_duplicates(features.columns)
    return tmp.drop(columns=['calendar_yearquarter'])


def add_city_features(features: pd.DataFrame, stats_city: pd.DataFrame):
    return features.merge(stats_city, on=[n.FIELD_CITY_CODE], how='left')


def build_store_location_stats_features(
        dates_when_predicting: list, stores: pd.DataFrame, stats_unemployment_zone: pd.DataFrame,
        stats_unemployment_department: pd.DataFrame, stats_population_city: pd.DataFrame,
        stats_income_city: pd.DataFrame):

    # Add iris_code_commune of store
    features = get_all_store_dates_combinations(dates_when_predicting, stores, False)

    # Find corresponding quarter
    # Assumption: statistics for one quarter are only published the quarter after it happened
    features['quarter'] = features['date_when_predicting'].apply(convert_month_to_quarter)
    features['quarter'] = features['quarter'].apply(partial(misc.substract_period, add=1, highest_period=4))
    features['department_code'] = features[n.FIELD_CITY_CODE].astype(str).str[:2]

    # Unemployment stats per employment zone & department
    def remove_non_numeric_codes(df, col):
        df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
        df[col] = df[col].astype(int)

        if n.FIELD_CITY in df:
            df.drop(columns=[n.FIELD_CITY], inplace=True)
        return df

    stats_unemployment_zone = remove_non_numeric_codes(stats_unemployment_zone, n.FIELD_CITY_CODE)
    features = add_latest_available_quarterly_statistics(features, stats_unemployment_zone, [n.FIELD_CITY_CODE], 'city')
    features = add_latest_available_quarterly_statistics(features, stats_unemployment_department, ['department_code'])

    # Add population per city
    stats_population_city = remove_non_numeric_codes(stats_population_city, n.FIELD_CITY_CODE)
    features = add_city_features(features, stats_population_city)

    # Add income per city
    stats_income_city = remove_non_numeric_codes(stats_income_city, n.FIELD_CITY_CODE)
    features = add_city_features(features, stats_income_city)

    # ADD PIPELINE TO FILL NA
    return features.drop(columns=[n.FIELD_CITY_CODE, 'quarter', 'department_code'])
