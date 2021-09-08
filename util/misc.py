"""
This module contains miscellaneous functions that are standalone enough to not
fit into any submodule.
"""

import datetime
import pandas as pd
import numpy as np

pd.set_option('chained_assignment', None)


def memory_usage(pandas_obj):
    """
    Calculate memory storage for Pandas Objects
    """
    if isinstance(pandas_obj, pd.DataFrame):
        usage = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage = pandas_obj.memory_usage(deep=True)
    usage_conv = usage / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_conv)


def downcast_numeric(pandas_obj, my_type):
    """
    Downcast numeric types to same memory
    """
    assert isinstance(pandas_obj, pd.DataFrame), "{0} is not a Pandas DataFrame".format(pandas_obj)
    assert isinstance(my_type, str), "Your type name {0} was not provided as a String".format(my_type)
    downcast_num_type = pandas_obj.select_dtypes(include=[my_type])
    converted_num_type = downcast_num_type.apply(pd.to_numeric, downcast='unsigned')
    return converted_num_type


def convert_categorical(pandas_obj, is_training=True):
    """
    Convert object type to category type

    Args:
        is_training (bool): Argument is added here because while testing we don't want to check whether the variable
        is categorical or not. Proportions may not be the same in the test & train sets.
    """
    assert isinstance(pandas_obj, pd.DataFrame), "{0} is not a Pandas DataFrame".format(pandas_obj)
    for key in pandas_obj.select_dtypes(include=['object']).keys():
        num_unique_values = pandas_obj[key].nunique()
        num_total_values = len(pandas_obj[key])
        if num_total_values != 0:
            if num_unique_values / num_total_values < 0.33 or not is_training:
                pandas_obj.loc[:, key] = pandas_obj[key].astype('category')
    return pandas_obj


def optimize_memory(pandas_obj):
    """
    Optimize memory for numeric types
    """
    assert isinstance(pandas_obj, pd.DataFrame), "{0} is not a Pandas DataFrame".format(pandas_obj)
    if isinstance(pandas_obj,pd.DataFrame):
        print('memory usage before dtype downcast/conversion: ', memory_usage(pandas_obj))
        converted_int = downcast_numeric(pandas_obj, 'int')
        converted_float = downcast_numeric(pandas_obj, 'float')
        pandas_obj[converted_int.columns] = converted_int
        pandas_obj[converted_float.columns] = converted_float
        # pandas_obj = convert_categorical(pandas_obj)
        print('memory usage after dtype downcast/conversion: ', memory_usage(pandas_obj))
        return pandas_obj


def get_year_week(date):
    """
    Returns the year_week value for a given date.
    We consider that the year week corresponds to the Iso-Code of the next
    saturday to match the year weeks used in the stock table.
    Input: datetime object or str
    Returns: date in YYYYWW format
    """
    if isinstance(date, str) or isinstance(date, datetime.date):
        date = pd.to_datetime(date)

    return datetime.datetime.strftime(date, "%G%V")


def year_week_offset(year_week, **offset_kwargs):
    """
    Inputs:
        - year_week: str YYYYWW
        - offset_kwargs: variables to pd.DateOffset()
    Returns:
        - str YYYYMM
    Examples:
        - year_week_offset("201801", days=7)    -> '201802'
        - year_week_offset("201801", months=6)  -> '201826'
        - year_week_offset("201801", months=-6) -> '201727'
    """
    year_week_date = get_datetime_from_year_week(year_week)
    return get_year_week(year_week_date + pd.DateOffset(**offset_kwargs))


def get_datetime_from_year_week(year_week: str, dow: str = '6') -> datetime:
    """
    Inputs:
        year_week: date in year-week format
        dow: day of the week to which to convert date time (default: Sunday)
    Returns:
        date in YYYYMMDD format
    """
    # pandas.to_datetime does not work with %G%v (Iso format), which is why we have to
    # use strptime
    return pd.to_datetime(datetime.datetime.strptime(year_week + '-' + dow, '%G%V-%w'))


def get_year_week_range(first, last):
    """
    Returns a range of year weeks.
    """

    first_date = get_datetime_from_year_week(first)
    end_date   = get_datetime_from_year_week(last)

    all_year_weeks = [get_year_week(_) for _ in pd.date_range(start=first_date, end=end_date, freq='W-SAT')]
    return all_year_weeks


def year_week_diff(first, second):
    """
    Returns the difference in year_weeks as an int
    """

    def to_date(year_week):
        """
        We aren't using get_datetime_from_year_week() because it cannot be vectorized
        (since pandas.to_datetime does not use ISO format, and we end up using the
        datetime module) and we don't want to use a .apply()
        It does not matter here, because the same rule is applied to both year_weeks,
        so the difference in dates is going to be the same.
        """
        return pd.to_datetime(year_week + "-0", format="%Y%U-%w")

    diff = to_date(first) - to_date(second)

    return diff / np.timedelta64(1, 'W')


def get_datetime_from_year_month(year_month: str) -> datetime:
    """
    Inputs:
        year_month: date in year-week format
    Returns:
        date in YYYYMMDD format
    """

    return pd.to_datetime(year_month + '01', format='%Y%m%d')


def get_year_month_range(first, last):
    """
    :param first:
    :param last:
    :return:
    """
    first_date = get_datetime_from_year_month(first)
    end_date = get_datetime_from_year_month(year_month_offset(last, months=1))

    date_range = pd.date_range(start=first_date, end=end_date, freq='M')

    all_year_months = [get_year_month(_) for _ in date_range]
    return all_year_months


def get_year_month(date):
    """
    :param date:
    :return:
    """
    return str(date.year) + str(date.month).zfill(2)


def year_month_offset(year_month, **offset_kwargs):
    """
    Inputs:
        - year_month: str YYYYMM
        - offset_kwargs: variables to pd.DateOffset()
    Returns:
        - str YYYYMM
    """
    year_month_date = get_datetime_from_year_month(year_month)
    return get_year_month(year_month_date + pd.DateOffset(**offset_kwargs))


def get_year(year_week):
    """
    Works with both year_week and year_month
    Inputs:
        - year_week: pd.Series
    Returns:
        - year: pd.Series
    """
    return year_week.astype(str).str[:4].astype(int)


def create_list_period(start, end, is_weekly_mode: bool=True):
    """
    This function returns the list of date between start and end
    exemple usage : create_list_period(201801, 201852)
    Inputs:
        - start: integer indicating the first date to return
        - end: integer indicating the last date to return
        -is_weekly_mode: boolean that indicates if start and end are indicating weeks or months
    Returns:
        - res: the list of date between start and end
    """
    res = list()
    if is_weekly_mode:  # Weekly mode
        while start <= end:
            res.append(start)
            if start % 100 != 52:
                start += 1
            else:
                start = ((start//100)+1)*100+1
        return res
    else:  # Monthly mode
        while start <= end:
            res.append(start)
            if start % 100 != 12:
                start += 1
            else:
                start = ((start//100)+1)*100+1
        return res


def diff_period(whenpredic, topredic, is_weekly: bool=True):
    """
    This function returns the number of weeks (or months) between whenpredic and topredic
    Inputs:
        - whenpredic: integer indicating the smallest of the two dates
        - topredic: integer indicating the biggest of the two dates
        - is_weekly_mode: boolean that indicates if whenpredic and topredic are indicating weeks or months
    Returns:
        - res: the number of weeks between whenpredic and topredic
    """
    highest_period = 52 if is_weekly else 12
    res = 0
    while whenpredic < topredic:
        res += 1
        if whenpredic % 100 != highest_period:
            whenpredic += 1
        else:
            whenpredic = ((whenpredic//100)+1)*100+1
    return res


def get_nbsales_zeros(df):
    """
    This function returns a series containing the number of most recent weeks with no sales
    Inputs:
        - df: a dataframe with weekly sales (each column is a week)
    Returns:
        - res: the number of recent weeks where sales = 0
    """
    t = (1-df.mask(df > 0, 1))

    def sum_series_zeros(res, cond, new):
        toadd = new * cond
        res2 = res + toadd
        cond2 = cond * new
        return res2, cond2
    clsr = t.columns.values
    res = t[clsr[-1]]
    cond = t[clsr[-1]]
    for i in range(len(clsr)-1, 0, -1):
        res, cond = sum_series_zeros(res, cond, t[clsr[i-1]])
    return res


def add_period(date, add, highest_period: int=52):
    """
    This function returns a week (or month) equal to date + add
    Inputs:
        - date: integer indicating a week number or month number
        - add: integer indicating the number of weeks to add
        - is_weekly_mode: boolean that indicates if the date is in week or months
    Returns:
        - res: the resulting operation (a new date, integer)
    """
    i = 0
    while i < add:
        if date % 100 != highest_period:
            date += 1
        else:
            date = ((date//100)+1)*100+1
        i += 1
    return date


def substract_period(date, add, highest_period: int=52):
    i = 0
    while i < add:
        if date % 100 != 1:
            date -= 1
        else:
            date = ((date//100)-1)*100 + highest_period
        i += 1
    return date


def delta_between_periods(period_a, period_b, is_weekly: bool=True):
    if period_a > period_b:
        return diff_period(period_b, period_a, is_weekly)
    return -diff_period(period_a, period_b, is_weekly)


def get_all_datehorizons(ldtp, fh, is_weekly: bool = True):
    highest_period = 52 if is_weekly else 12
    dtp = list()
    for i in fh:
        for j in ldtp:
            dtp.append(add_period(j, i, highest_period=highest_period))
    return dtp


def cross_join(df1, df2, suffix='_x'):
    df1['key'] = 1
    df2['key'] = 1
    df = df1.merge(df2, on='key', how='outer', suffixes=['', suffix]).drop(columns=['key'])
    df1.drop(columns=['key'], inplace=True)
    df2.drop(columns=['key'], inplace=True)
    return df
