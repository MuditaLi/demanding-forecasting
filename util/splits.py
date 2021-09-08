"""
Script to get split of weekly forecast per month
Example: split = split_monthly_forecast_to_weekly(
    [12, 18], pd.to_datetime(pd.Timestamp.today().date()) - np.timedelta64(1, 'Y'))
"""
import pandas as pd
import numpy as np
from functools import partial
import preprocessor.names as n
from util.misc import diff_period


def cast_days(days):
    if not isinstance(days, pd.DatetimeIndex):
        return pd.DatetimeIndex(list(days))
    return days


def get_offset_date(offset_month: int, start_date):
    return start_date + pd.offsets.MonthBegin(offset_month)


def get_week_number(days):
    days = cast_days(days)
    year = days.year  # - np.timedelta64(1, 'D')
    year = year.where((days.month < 12) | (days.week >= 40), year + 1)  # correct date for 1st week of the year

    return year.astype(str).str.zfill(2) + days.week.astype(str).str.zfill(2)


def get_month_nunber(days):
    days = cast_days(days)
    return days.year.astype(str).str.zfill(2) + days.month.astype(str).str.zfill(2)


def get_week_month_ratio(days, beginning_month: bool=False):
    """
    Function to split a weekly forecast depending on default technical periods, i.e. working days.
    Ratio computed allows to determine the contribution of a given week to each month it belongs to.

    Assumption: Bank holidays are not filtered out for now. Simple computation based on all week days.
    """
    months = get_month_nunber(days)
    weeks = get_week_number(days)

    if beginning_month:
        ratio = np.clip(1 - (days.dayofweek % 7) / 5, 0, None)  # 5 = Saturday | 6 = Sunday
    else:
        ratio = np.clip((days.dayofweek % 7 + 1) / 5, None, 1)

    return pd.DataFrame(list(zip(months, weeks, ratio, (days.day == 1).astype(int))),
                        columns=[n.FIELD_YEARMONTH, n.FIELD_YEARWEEK, 'ratio', 'first_day_of_month'])


def compute_technical_periods_monthly_split(horizons, date_when_predicting):
    """
    Method to split our weekly forecast for APO feed. Some weeks can belong to two months at the same time, so we need
    to compute the contribution of our weekly forecast to a given month. Simple technical periods logic
    based on weekdays (ignoring bank holidays).

    Args:
        horizons (list): [minimum horizon in months (int), maximum horizon in months (int)]
        min_date_when_predicting (pd.Timestamp or int): date when you do the forecast. Datetime or week number

    Returns:
        Table
    """
    if isinstance(date_when_predicting, int):
        # Extract Monday timestamp from week number
        date_when_predicting = pd.to_datetime(str(date_when_predicting) + ' w1', format='%Y%W w%w')

    if date_when_predicting is None:
        date_when_predicting = pd.to_datetime(pd.Timestamp.today().date())

    get_date = partial(get_offset_date, start_date=date_when_predicting)
    first_days = pd.date_range(start=get_date(horizons[0] - 1), end=get_date(horizons[1] + 1), freq='MS')
    last_days = pd.date_range(start=get_date(horizons[0] - 1), end=get_date(horizons[1] + 2), freq='MS')
    last_days -= np.timedelta64(1, 'D')

    partial_weeks = [get_week_month_ratio(first_days, beginning_month=True),
                     get_week_month_ratio(last_days, beginning_month=False)]
    partial_weeks = pd.concat(partial_weeks, axis=0).sort_values([n.FIELD_YEARMONTH, n.FIELD_YEARWEEK])

    all_weeks = pd.date_range(start=get_date(horizons[0] - 1), end=get_date(horizons[1] + 2), freq='W')
    all_weeks = zip(get_month_nunber(all_weeks), get_week_number(all_weeks))
    all_weeks = pd.DataFrame(list(all_weeks), columns=[n.FIELD_YEARMONTH, n.FIELD_YEARWEEK])

    split = (pd.concat([partial_weeks, all_weeks], axis=0, sort=False)
             .drop_duplicates([n.FIELD_YEARWEEK, n.FIELD_YEARMONTH])
             .sort_values([n.FIELD_YEARWEEK, n.FIELD_YEARMONTH])
             .fillna(1)
             )

    split[[n.FIELD_YEARMONTH, n.FIELD_YEARWEEK]] = split[[n.FIELD_YEARMONTH, n.FIELD_YEARWEEK]].astype(int)

    return (split
            .drop(columns=['first_day_of_month'])
            .sort_values([n.FIELD_YEARMONTH, n.FIELD_YEARWEEK])
            )


def get_columns_scenarios(df):
    cols = [col for col in df.columns if col.startswith('temperature_w0')]
    cols = get_columns_predictions(df).union(cols)
    return cols


def get_columns_predictions(df):
    """
    Args:
        df (pd.DataFrame): data frame containing the predictions

    Returns:
        list of columns that need to be split between months
    """
    cols = [col for col in df.columns if col.startswith(n.FIELD_PREDICTION)]
    cols += ['ml_baseline_wo_promo', 'ml_promo_uplift']
    return set(cols).intersection(df.columns)


def split_monthly_forecast_to_weekly(date_when_predicting: int, predictions: pd.DataFrame,
                                     col_predictions: set([str])=set([n.FIELD_PREDICTION])):
    """
    Function to automatically split the predicted demand between months (for weeks which have a common denominator)

    Args:
        date_when_predicting (int): Week or month number of when we do the predictions
        predictions (pd.DataFrame): output of the ML model
        col_predictions (str): name of column with the predicted sales volume

    Returns:
        Data frame with split sales

    """
    horizons = predictions['date_to_predict'].agg([min, max]).values
    horizons = list(map(lambda x: diff_period(date_when_predicting, x, is_weekly=True) // 4, horizons))
    horizons[1] += 1  # convert horizon in month & ensure that the scope of the split is exhaustive vs predicted weeks

    split = compute_technical_periods_monthly_split(horizons, date_when_predicting)
    predictions = (predictions
                   .rename(columns={'date_to_predict': n.FIELD_YEARWEEK})
                   .merge(split, on=n.FIELD_YEARWEEK, how='left')
                   )

    for col in col_predictions:
        predictions[col] = predictions[col] * predictions['ratio']

    return predictions.drop(columns=['ratio'])
