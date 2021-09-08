import preprocessor.names as n
import pandas as pd


def features_open_orders(hfa, granularity, dates_when_predicting, dates_to_predict):
    """
    """
    hfa_period = hfa[hfa['load_week'].isin(dates_when_predicting)]
    temp = hfa_period.groupby(['load_week'] + granularity)[
        n.FIELD_OPEN_ORDERS].sum().reset_index()

    temp = temp[temp['calendar_yearweek'].isin(dates_to_predict)]
    temp.rename(columns={'load_week': 'date_when_predicting'}, inplace=True)
    temp.rename(columns={'calendar_yearweek': 'date_to_predict'}, inplace=True)

    return temp


def feature_sumopen_orders(hfa, granularity, dates_when_predicting, dates_to_predict):
    """
    Sum of open orders from date when predicting until date to predict
    """
    hfa_period = hfa[hfa['load_week'].isin(dates_when_predicting)]
    temp = hfa_period.groupby(['load_week'] + granularity)[
        n.FIELD_OPEN_ORDERS].sum().reset_index()

    temp = temp[temp['calendar_yearweek'].isin(dates_to_predict)]
    temp.rename(columns={'load_week': 'date_when_predicting'}, inplace=True)
    temp.rename(columns={'calendar_yearweek': 'date_to_predict'}, inplace=True)

    res = pd.DataFrame()

    for w in temp['date_when_predicting'].unique():
        orders_known_atweek = temp[temp['date_when_predicting'] == w].copy()
        orders_known_atweek['open_orders'] = orders_known_atweek['open_orders'].fillna(0)

        sumvalues = orders_known_atweek.groupby(['plant', 'customer_planning_group', 'lead_sku'])[
            'open_orders'].cumsum()
        orders_known_atweek['open_orders'] = sumvalues.values
        orders_known_atweek.rename(columns={'open_orders': 'sumopen_orders'}, inplace=True)
        orders_known_atweek['date_when_predicting'] = w
        res = pd.concat([res, orders_known_atweek])

    return res
