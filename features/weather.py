import re
import math
import numpy as np
import pandas as pd
import preprocessor.names as n
import itertools
from functools import partial
from util import misc


def distance(lat1, lon1, lat2, lon2):
    """
    Find closest weather city for each plant
    NOTE: I looked a bit online and the lat/lon most common distance measure seems to be using Haversine's formula:
    - https://www.movable-type.co.uk/scripts/latlong.html?from=48.64703,-122.26324&to=48.6721,-122.265
    """
    radius = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


# Fill NaN
def fill_weather_na(data, key=n.FIELD_PLANT_ID):
    # Fills first with previous value (of the right plant), if none found takes next value, if none found takes the mean
    return data.groupby(key).ffill().bfill().fillna(data.mean())


# Create windows range of lookup around dates_to_predict:
def generate_weather_windows(df, week_ranges, location_key=n.FIELD_PLANT_ID, timeframe_key=n.FIELD_YEARWEEK):
    # [c for c in list(df.columns) if c not in columns_to_ignore]
    original_columns = df.columns.difference([location_key, timeframe_key])  # list columns to create
    df = df.sort_values(by=[location_key, timeframe_key])  # Convenient to use shift function

    label = '_week' if (timeframe_key == n.FIELD_YEARWEEK) else '_month'
    for i in week_ranges:
        for col in original_columns:
            df[col + label + str(i)] = df.groupby([location_key])[col].shift(periods=(-1 * i))
    df = df.drop(original_columns, axis=1)
    return df


# Main function - sell-in model
def features_weather_sell_in(df_weather, plants_df, dates_to_predict, noise_lvl=0.01):
    """
    Generates the weather features. Output is on plant and date_to_predict (in year_week) level.
    : param df_weather: Cleaned weather (including geo loc)
    : param plant_df: Cleaned plants (including geo loc)
    : param dates_to_predict: List of dates at which the demand is predicted (date of deman). Can be int of str.
    : param noise_lvl: How much noise to be added. Noise is Gaussian dist around the feature columns value.
                       It is then multiplied by this number.
                       If noise_lvl = 1  -> The added noise will have a normal distribution
                       If noise_lvl = 0  -> No noise added
                       If noise_lvl > 1  -> Increase noise effect by that factor
                       If 0 < noise_lvl between < 1  -> Decrease noise by that factor
    """
    df_weather[n.FIELD_YEARWEEK] = list(map(misc.get_year_week, df_weather['time']))

    # Find closest weather_city to each plant
    df_cities = df_weather.drop_duplicates([n.FIELD_CITY])[[n.FIELD_CITY, n.FIELD_LATITUDE, n.FIELD_LONGITUDE]].copy().reset_index(drop=True)
    mapped_cities = []
    for index, row in plants_df.iterrows():   # Loops over every plant
        temp_df = df_cities.copy()
        temp_df['dist'] = [
            distance(row[n.FIELD_LATITUDE], row[n.FIELD_LONGITUDE],
                     temp_df[n.FIELD_LATITUDE][i], temp_df[n.FIELD_LONGITUDE][i]) for i in range(len(temp_df))]

        mapped_cities.append(temp_df.loc[temp_df['dist'].idxmin()][n.FIELD_CITY])   # Closest city

    plants_to_closest_city = pd.DataFrame(
        data={n.FIELD_PLANT_ID: plants_df[n.FIELD_PLANT_ID], n.FIELD_CITY: mapped_cities})

    # Create weather table
    mapped_df = pd.DataFrame()
    for index, row in plants_to_closest_city.iterrows():
        temp_df = df_weather[df_weather[n.FIELD_CITY] == row[n.FIELD_CITY]]
        temp_df[n.FIELD_PLANT_ID] = row[n.FIELD_PLANT_ID].strip()
        mapped_df = mapped_df.append(temp_df)

    # Groupby on a weekly basis - TODO - keep thinking about which data we want to include.
    mapped_df = mapped_df.groupby([n.FIELD_PLANT_ID, n.FIELD_YEARWEEK]).agg({'apparentTemperatureHigh': ['mean'],
                                                                             # 'cloudCover': ['mean'],
                                                                             # 'precipIntensity': ['sum'],
                                                                             # 'windSpeed': ['max'],
                                                                             # 'humidity':['mean'],
                                                                             # 'precipAccumulation': ['max']
                                                                             })

    mapped_df.columns = map('_'.join, mapped_df.columns.ravel())
    mapped_df = mapped_df.rename(index=str, columns={'apparentTemperatureHigh_mean': 'apparent_temperature_mean',
                                                     # 'cloudCover_mean': 'cloud_cover_mean',
                                                     # 'precipIntensity_sum': 'precipitations_agg',
                                                     # 'windSpeed_max': 'wind_speed_max',
                                                     # 'precipAccumulation_max': 'snow_cover_max'
                                                     })

    mapped_df = mapped_df.reset_index()

    # Create windows range of lookup around dates_to_predict
    mapped_df = generate_weather_windows(mapped_df, [-2, -1, 0, 1, 2])

    # Fill NaNs
    mapped_df = fill_weather_na(mapped_df)

    # Add noise
    for col in mapped_df.columns.difference([n.FIELD_PLANT_ID, n.FIELD_YEARWEEK]):
        std = mapped_df[col].std()
        noise = np.random.normal(0, std, len(mapped_df)) * noise_lvl  # noise_lvl could also be multiplied to std
        if mapped_df[col].min() >= 0:
            mapped_df[col] = mapped_df[col] + noise
            mapped_df[col] = mapped_df[col].clip(lower=0)
        else:
            mapped_df[col] = mapped_df[col] + noise

    # Filter for relevant dates
    dates_to_pred = map(str, dates_to_predict)
    mapped_df = mapped_df[mapped_df[n.FIELD_YEARWEEK].isin(dates_to_pred)].reset_index(drop=True)
    mapped_df[n.FIELD_YEARWEEK] = mapped_df[n.FIELD_YEARWEEK].astype(int)

    return mapped_df.rename(columns={n.FIELD_YEARWEEK: 'date_to_predict'})


# Main function - Sell-out model
def features_weather_sellout(df_weather, df_stores, df_cities, dates_to_predict, monthly_fcst=True, noise_lvl=0.01):
    """
    Generates the weather features. Output is on plant and date_to_predict (in year_week) level.
    : param df_weather: Cleaned weather (including geo loc)
    : param df_stores: consolidated_stores_list_scoped.csv
    : param df_cities: insee_data_cities_cleaned.csv
    : param dates_to_predict: List of dates at which the demand is predicted (date of deman). Can be int of str.
    : param noise_lvl: How much noise to be added. Noise is Gaussian dist around the feature columns value.
                       It is then multiplied by this number.
                       If noise_lvl = 1  -> The added noise will have a normal distribution
                       If noise_lvl = 0  -> No noise added
                       If noise_lvl > 1  -> Increase noise effect by that factor
                       If 0 < noise_lvl between < 1  -> Decrease noise by that factor
    """
    # Keep only cities where there is at least one store located
    df_stores = (df_stores[df_stores[n.FIELD_STORE_ID].notnull()]
                 .filter([n.FIELD_STORE_ID, n.FIELD_CITY_CODE]))
    df_cities = (df_cities[df_cities[n.FIELD_CITY_CODE].isin(df_stores[n.FIELD_CITY_CODE].unique())]
                 .filter([n.FIELD_CITY_CODE, n.FIELD_LATITUDE, n.FIELD_LONGITUDE]))

    # Create time column
    if monthly_fcst:
        timeframe_key, time_fct, windows_ranges = n.FIELD_YEARMONTH, misc.get_year_month, [-1, 0, 1]
    else:
        timeframe_key, time_fct, windows_ranges = n.FIELD_YEARWEEK, misc.get_year_week, [-2, -1, 0, 1, 2]

    df_weather['time'] = pd.to_datetime(df_weather['time'])
    df_weather[timeframe_key] = list(map(time_fct, df_weather['time']))
    df_weather[timeframe_key] = df_weather[timeframe_key].astype(int)

    # Find closest weather city for each store
    df_cities_weather = (df_weather
                         .drop_duplicates([n.FIELD_CITY])
                         .filter([n.FIELD_CITY, n.FIELD_LATITUDE, n.FIELD_LONGITUDE])
                         .reset_index(drop=True))

    mapped_cities = []
    for index, row in df_cities.iterrows():  # Loops over every store
        df_cities_weather['dist'] = list(map(
            lambda x: distance(row[n.FIELD_LATITUDE], row[n.FIELD_LONGITUDE], x[0], x[1]),
            df_cities_weather[[n.FIELD_LATITUDE, n.FIELD_LONGITUDE]].itertuples(index=False)
        ))
        mapped_cities.append(df_cities_weather.loc[df_cities_weather['dist'].idxmin()][n.FIELD_CITY])  # Closest city

    # match closest_city_weather = f(city code)
    df_cities = (df_cities
                 .filter([n.FIELD_CITY_CODE])
                 .merge(pd.DataFrame({n.FIELD_CITY_CODE: df_cities[n.FIELD_CITY_CODE], n.FIELD_CITY: mapped_cities}),
                        on=n.FIELD_CITY_CODE, how='left')
                 )

    # Groupby df_weather according to the timeframe
    agg_fcts = ['max', 'min', 'mean']
    df_weather.columns = list(map(str.lower, map(partial(re.sub, '([A-Z])', r'_\1'), df_weather.columns)))
    df_weather['precip_accumulation'] = df_weather['precip_accumulation'].fillna(0)
    df_weather = (df_weather
                  .rename(columns={'precip_accumulation': 'precipitations'})
                  .groupby([n.FIELD_CITY, timeframe_key])
                  .agg({'apparent_temperature_high': agg_fcts,
                        # 'apparent_temperature_low': agg_fcts,
                        'pressure': agg_fcts,
                        'precipitations': ['mean', 'max']})
                  .reset_index()
                  )
    df_weather.columns = map(lambda x: x.strip('_'), map('_'.join, df_weather.columns.ravel()))

    # Create windows range of lookup around dates_to_predict
    df_weather = generate_weather_windows(df_weather, windows_ranges,
                                          location_key=n.FIELD_CITY, timeframe_key=timeframe_key)

    # Fill NaNs
    df_weather = fill_weather_na(df_weather, key=n.FIELD_CITY)

    # Add noise
    for col in df_weather.columns.difference([n.FIELD_CITY, timeframe_key]):
        std = df_weather[col].std()
        noise = np.random.normal(0, std, len(df_weather)) * noise_lvl  # noise_lvl could also be multiplied to std
        if df_weather[col].min() >= 0:
            df_weather[col] = df_weather[col] + noise
            df_weather[col] = df_weather[col].clip(lower=0)
        else:
            df_weather[col] = df_weather[col] + noise

    df_weather_full = (pd.DataFrame(list(itertools.product(df_stores[n.FIELD_STORE_ID].unique(), dates_to_predict)),
                                    columns=[n.FIELD_STORE_ID, timeframe_key])
                       .merge(df_stores, on=n.FIELD_STORE_ID, how='left')
                       .merge(df_cities, on=n.FIELD_CITY_CODE, how='left')
                       .drop(columns=[n.FIELD_CITY_CODE])
                       )
    df_weather_full[timeframe_key] = df_weather_full[timeframe_key].astype(int)
    df_weather_full = (df_weather_full
                       .merge(df_weather, on=[n.FIELD_CITY, timeframe_key], how="left")
                       .drop(columns=[n.FIELD_CITY])
                       )

    return df_weather_full.rename(columns={timeframe_key: 'date_to_predict'})
