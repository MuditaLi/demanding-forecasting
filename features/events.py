import itertools
import numpy as np
import pandas as pd
from functools import partial
import preprocessor.names as n
import util.misc as misc
from features.weather import distance
from features.zone_stats import get_all_store_dates_combinations


def assign_to_closest_plant(df_events, df_plants_location):

    # Since F001 and F015 are too close to each other, we keep one in selection
    df_plants_location = df_plants_location[df_plants_location[n.FIELD_PLANT_ID] != 'F015']

    # find closest plant (assumption customer gets supply for a given store at the closest possible plant).
    df_events_plant = misc.cross_join(df_events, df_plants_location, suffix='_plant')
    columns = [n.FIELD_LATITUDE, n.FIELD_LONGITUDE, n.FIELD_LATITUDE + '_plant', n.FIELD_LONGITUDE + '_plant']
    df_events_plant['distance'] = df_events_plant[columns].apply(lambda x: distance(*x), axis=1)

    df_events_plant = (df_events_plant
                       .sort_values('distance', ascending=True)
                       .drop_duplicates(df_events.columns)
                       .filter(list(df_events.columns) + [n.FIELD_PLANT_ID])
                       .drop(columns=[n.FIELD_LATITUDE, n.FIELD_LONGITUDE])
                       )

    tmp = df_events_plant[df_events_plant[n.FIELD_PLANT_ID] == 'F001']
    tmp[n.FIELD_PLANT_ID] = tmp[n.FIELD_PLANT_ID].replace('F001', 'F015')
    df_events_plant = pd.concat([df_events_plant, tmp], axis=0, sort=False)  # Add back df15

    return df_events_plant


def build_sell_in_events_features(df_events: pd.DataFrame, df_plants_location: pd.DataFrame, dates_to_predict: list,
                                  dates_when_predicting: list, horizon_weeks: int=4):

    df_features = pd.DataFrame(list(map(lambda x: tuple([x[0]] + list(x[1])),
                                        itertools.product(df_plants_location[n.FIELD_PLANT_ID].unique(),
                                                          zip(dates_when_predicting, dates_to_predict)))),
                               columns=[n.FIELD_PLANT_ID, 'date_when_predicting', 'date_to_predict'])

    df_events[n.FIELD_YEARWEEK + '_start_corrected'] = df_events[n.FIELD_YEARWEEK + '_start'].apply(
        partial(misc.substract_period, add=horizon_weeks, highest_period=52))

    df_events_plant = assign_to_closest_plant(df_events, df_plants_location)

    features = {
        'holiday': 'max',
        'festival': 'sum',
        'other_sport_events': 'sum',
        'major_sport_events': 'sum',
        'rank': ['max', 'mean'],
        'local_rank': ['mean']
    }

    df_features = df_features.merge(df_events_plant, on=[n.FIELD_PLANT_ID], how='left')
    df_features = df_features[
        (df_features[n.FIELD_YEARWEEK + '_end'] >= df_features['date_to_predict'])
        &
        (df_features[n.FIELD_YEARWEEK + '_start_corrected'] <= df_features['date_to_predict'])
    ]

    # Only apply filter on first_seen (time at which event was added to data set) for 2018 onwards data
    df_features['filter'] = \
        (df_features[n.FIELD_YEARWEEK + '_first_seen'] >= df_features['date_when_predicting']).astype(int)
    df_features['filter'] = df_features['filter'].where(df_features[n.FIELD_YEARWEEK + '_start'] // 100 > 2018, 1)
    df_features['filter'] = df_features[df_features['filter'] == 1]

    df_features = (df_features
                   .groupby([n.FIELD_PLANT_ID, 'date_when_predicting', 'date_to_predict'])
                   .agg(features)
                   .reset_index()
                   )
    df_features.columns = map(lambda x: x.strip('_'), map('_'.join, df_features.columns.ravel()))
    return df_features


def find_number_of_event_days_in_month(df_features):
    """
    Since binary holiday indicator is not that relevant at monthly granularity, we will look at number of holidays
    in month instead.
    """
    # Check that date to predict is a month
    assert (df_features['date_to_predict'] % 100).max() <= 12

    for col in ['start', 'end']:
        df_features[col] = pd.to_datetime(df_features[col])

    df_features['beg_holiday'] = pd.to_datetime(df_features['date_to_predict'], format='%Y%m')
    df_features['end_holiday'] = df_features['beg_holiday'] + pd.offsets.MonthBegin(1) - np.timedelta64(1, 'D')
    df_features['end_holiday'] = df_features[['end', 'end_holiday']].min(axis=1)
    df_features['beg_holiday'] = df_features[['start', 'beg_holiday']].max(axis=1)
    df_features['days'] = (df_features['end_holiday'] - df_features['beg_holiday']) / np.timedelta64(1, 'D') + 1

    # Holidays should not be considered as so if not in same month
    df_features['holiday'] = df_features['holiday'].where(df_features['days'] < 1, 0)

    # Remove field of all non holidays events
    df_features['days'] = df_features['days'].where(df_features['holiday'] == 1, 0)

    return df_features.drop(columns=['beg_holiday', 'end_holiday'])


def build_sell_out_events_features(dates_to_predict: list, dates_when_predicting: list,
                                   stores: pd.DataFrame, events: pd.DataFrame,
                                   cities: pd.DataFrame, max_distance_km: int=50, horizon_months: int=1):

    df_features = get_all_store_dates_combinations(dates_when_predicting, stores, False, dates_to_predict)

    # scope data (relevant columns)
    stores = (stores
              .merge(cities, on=n.FIELD_CITY_CODE)
              .filter([n.FIELD_STORE_ID, n.FIELD_CITY_CODE, n.FIELD_LATITUDE, n.FIELD_LONGITUDE])
              .drop_duplicates())

    events = events.drop(columns=filter(lambda x: x.startswith(n.FIELD_YEARWEEK), events.columns))
    events = events[events[n.FIELD_YEARMONTH + '_end'] > 201601]
    # events = events.reset_index(drop=False).rename(
    #     columns={'index': 'event_id', n.FIELD_LATITUDE: n.FIELD_LATITUDE + '_event',
    #              n.FIELD_LONGITUDE: n.FIELD_LONGITUDE + '_event'}
    # )
    events = events.rename(columns={n.FIELD_LATITUDE: n.FIELD_LATITUDE + '_event',
                                    n.FIELD_LONGITUDE: n.FIELD_LONGITUDE + '_event'})

    # We want to take all events occurring the month we want to predict and the month after
    # (in case the event is at the beginning of next month)
    # Operation done before merge (faster)
    if horizon_months:
        events[n.FIELD_YEARMONTH + '_filter'] = events[n.FIELD_YEARMONTH + '_start'].apply(
            partial(misc.substract_period, add=horizon_months, highest_period=12))

    events_store = misc.cross_join(
        stores.filter([n.FIELD_LATITUDE, n.FIELD_LONGITUDE]).drop_duplicates(),
        events.filter([n.FIELD_LATITUDE + '_event', n.FIELD_LONGITUDE + '_event']).drop_duplicates(),
        suffix='_event'
    )
    relevant_columns = [n.FIELD_LATITUDE, n.FIELD_LONGITUDE, n.FIELD_LATITUDE + '_event', n.FIELD_LONGITUDE + '_event']
    events_store['distance'] = (events_store[relevant_columns].apply(lambda x: distance(*x), axis=1))
    events_store = events_store[events_store['distance'] <= max_distance_km]

    events_store = events_store.merge(events, on=[n.FIELD_LONGITUDE + '_event', n.FIELD_LATITUDE + '_event'], how='left')
    events_store = events_store.merge(stores.filter([n.FIELD_STORE_ID, n.FIELD_LATITUDE, n.FIELD_LONGITUDE]),
                                      on=[n.FIELD_LONGITUDE, n.FIELD_LATITUDE], how='left')

    events_store = events_store.drop(
        columns=filter(lambda x: x.startswith(n.FIELD_LATITUDE) or x.startswith(n.FIELD_LONGITUDE), events_store.columns)
    )

    # # Compute distance between event & store
    # events_store = np.array(list(itertools.product(stores[n.FIELD_STORE_ID], events['event_id'])))
    # distances = itertools.product(stores.filter([n.FIELD_LATITUDE, n.FIELD_LONGITUDE]).to_records(index=False),
    #                               events.filter([n.FIELD_LATITUDE, n.FIELD_LONGITUDE]).to_records(index=False))
    # distances = np.array(list(map(lambda x: distance(*x[0], *x[1]), distances)))
    #
    # # Keep events in specified area around a store
    # events_store = events_store[(distances <= max_distance_km).astype(bool)]
    # distances = np.extract((distances <= max_distance_km), distances)
    # events_store = pd.DataFrame(np.insert(events_store, 2, distances, axis=1),
    #                             columns=[n.FIELD_STORE_ID, 'event_id', 'distance'])
    # events_store['event_id'] = events_store['event_id'].astype(int)
    # events_store['distance'] = events_store['distance'].astype(float)
    # df_features = df_features.merge(events_store, on=n.FIELD_STORE_ID, how='left')
    # df_features = df_features.merge(events, on='event_id', how='left')

    # Merge to find relevant events for a given date to predict
    df_features = df_features.merge(events_store, on=n.FIELD_STORE_ID, how='left')

    df_features = df_features[
        (df_features['date_to_predict'] >= df_features[n.FIELD_YEARMONTH + '_filter'])
        &
        (df_features['date_to_predict'] <= df_features[n.FIELD_YEARMONTH + '_end'])
    ]

    # TODO - ADD FILTER BASED ON DATE_WHEN_PREDICTING - "first_seen" field does not seem reliable in the past

    features = {
        'holiday': 'max',
        'days': 'max',
        'festival': 'sum',
        'other_sport_events': 'sum',
        'major_sport_events': 'sum',
        'rank': ['max', 'mean'],
        'local_rank': ['mean']
    }

    # Holiday features for sell-out (binary not that relevant). Better to look at number of holidays.
    # Value in "days" column  can be negative because it could be in next month. In this case, the feature gives the
    # distance to the next event
    df_features = find_number_of_event_days_in_month(df_features)

    df_features = (df_features
                   .groupby([n.FIELD_STORE_ID, 'date_to_predict'])
                   .agg(features)
                   .reset_index()
                   )
    df_features.columns = map(lambda x: x.strip('_'), map('_'.join, df_features.columns.ravel()))
    return df_features
