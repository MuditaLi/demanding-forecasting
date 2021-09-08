import unicodedata
import numpy as np
import pandas as pd

import util.input_output as io
import preprocessor.names as n
from util.paths import PATH_DATA_RAW, PATH_DATA_FORMATTED


def get_week_number(days):
    # Carlsberg weeks start on Monday and end on Sunday, so we need to adjust the way pandas compute the week number
    return (days - np.timedelta64(1, 'D')).dt.week


def format_events_data(file_name, timezone):
    data = io.read_csv(PATH_DATA_RAW, 'Events', file_name)
    data[[n.FIELD_LATITUDE, n.FIELD_LONGITUDE]] = (data['location']
                                                   .str.strip('"')
                                                   .str.split(',', expand=True)
                                                   .astype(float)
                                                   )
    data = data[~data['category'].isin(['disasters', 'terror'])]
    data[[n.FIELD_ZIP_CODE, n.FIELD_CITY]] = data['venue_formatted_address'].str.extract(r'\n([0-9]{5})(\D*)\n')
    data.drop(columns=['location', 'country', 'id', 'state', 'duration', 'timezone',
                       'venue_formatted_address', 'description', 'venue_name'], inplace=True)

    relevant_columns = [
        'start',
        'end',
        n.FIELD_LATITUDE,
        n.FIELD_LONGITUDE,
        'category',
        'scope',
        'labels',
    ]
    data.loc[data['city'].notnull(), 'city'] = data.loc[data['city'].notnull(), 'city'].apply(
        lambda y: unicodedata.normalize('NFKD', y).encode('ASCII', 'ignore').decode())

    def format_datetime(serie):
        return pd.to_datetime(serie).dt.tz_localize(timezone, ambiguous=True).dt.tz_localize(None)

    for x in ['start', 'end', 'first_seen']:
        data[x] = format_datetime(data[x])
        year = data[x].dt.year

        data['correction'] = 1  # Account for dates that could be in first week of next year
        data['correction'] = data['correction'].where((data[x].dt.month >= 12) &
                                                      ((get_week_number(data[x]) == 1) |
                                                       (get_week_number(data[x]) == 53)), 0)

        year = (year + data['correction']).astype(str).str.replace('\.0', '')
        week = get_week_number(data[x]).astype(str).str.replace('\.0', '').str.zfill(2)
        week = week.where(week != '53', '01')  # Issue when we get week 53
        month = data[x].dt.month.astype(str).str.replace('\.0', '').str.zfill(2)

        data['_'.join([n.FIELD_YEARWEEK, x])] = year + week
        data['_'.join([n.FIELD_YEARMONTH, x])] = year + month

        for col in [n.FIELD_YEARWEEK, n.FIELD_YEARMONTH]:
            data['_'.join([col, x])] = (data['_'.join([col, x])].str.replace('nannan', str(np.nan)))

        data.drop(columns=['correction'], inplace=True)

    # data.drop(columns=['start', 'end', 'first_seen'], inplace=True)
    data.drop(columns=['first_seen'], inplace=True)

    l = list(map(lambda x: x.strip('"').split(','), data['labels'].unique()))
    labels = set()
    for x in l:
        for label in x:
            labels.add(label)

    len(labels)

    # data['labels'].str.strip('"').str.split(',', expand=True)
    for label in sorted(labels):
        data[label.replace('-', '_')] = data['labels'].str.contains(label).astype(int)

    data.drop(columns=['labels'], inplace=True)

    return data, labels


def clean_events_data():
    # Inputs
    file_name = 'fr_predicthq_export_end_dates_2015_2019.csv'
    timezone = 'Europe/Paris'
    events_threshold_importance = 65

    data, labels = format_events_data(file_name, timezone)
    data = data[data['rank'] >= events_threshold_importance]

    # Should definetly not keep labels that are highly correlated.
    cor = data[list(map(lambda x: x.replace('-', '_'), labels))].corr()
    for c in cor.columns:
        tmp = list(cor.loc[(cor[c] > 0.9), c].index.difference([c]))
        if tmp:
            print(c, tmp)

    columns_drop = ['school', 'outdoor', 'concert', 'social', 'movie', 'science', 'politics', 'marathon',
                    'attraction', 'pga', 'politics', 'fundraiser', 'health', 'community', 'comedy', 'technology',
                    'food', 'entertainment', 'club', 'business', 'education', 'conference', 'music',
                    'performing_arts', 'family']

    def combine_categories(category_name: str, categories: list, columns_drop):
        data[category_name] = (data[categories].sum(axis=1) > 0).astype(int)
        columns_drop += categories
        return columns_drop

    columns_drop = combine_categories('other_sport_events',
                                      ['american_football', 'auto_racing', 'badminton', 'baseball', 'boxing',
                                       'running', 'f1', 'hockey', 'horse_racing', 'volleyball'], columns_drop)

    columns_drop = combine_categories('major_sport_events',
                                      ['soccer', 'rugby', 'basketball', 'golf', 'tennis'], columns_drop)
    columns_drop = combine_categories('live_show', ['music', 'performing_arts'], columns_drop)
    columns_drop = combine_categories('brainy_events', ['business', 'education', 'conference'], columns_drop)

    data.drop(columns=set(columns_drop), inplace=True)

    # Correct discrepencies in holidays data
    for x in ['_start', '_end']:
        data[n.FIELD_YEARWEEK + x] = data[n.FIELD_YEARWEEK + x].astype(int)

    data.loc[data[n.FIELD_YEARWEEK + '_end'] - data[n.FIELD_YEARWEEK + '_start'] >= 90,
             n.FIELD_YEARWEEK + '_start'] += 100

    data = data[[
        'title',
        'scope',
        'rank',
        'local_rank',
        'latitude',
        'longitude',
        'start',
        'end',
        'calendar_yearweek_first_seen',
        'calendar_yearweek_start',
        'calendar_yearweek_end',
        'calendar_yearmonth_first_seen',
        'calendar_yearmonth_start',
        'calendar_yearmonth_end',
        'sport',
        'festival',
        'holiday',
        'other_sport_events',
        'major_sport_events',
    ]]
    data['filter'] = data['sport'] - (data[['other_sport_events', 'major_sport_events']].sum(axis=1) > 0).astype(int)
    data['other_sport_events'] = data[['other_sport_events', 'filter']].max(axis=1)
    data.drop(columns=['filter', 'sport'], inplace=True)
    data.sort_values(['rank'], inplace=True, ascending=False)

    io.write_csv(data, PATH_DATA_FORMATTED, 'events_clean_v2.csv')


if __name__ == '__main__':
    clean_events_data()
