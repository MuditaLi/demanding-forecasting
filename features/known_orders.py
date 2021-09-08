import preprocessor.names as n
import pandas as pd


def features_known_orders(hfa, granularity, dates_when_predicting, dates_to_predict):
    """
    Known orders are shipments already done for the newt weeks
    """
    hfa_period = hfa[hfa['load_week'].isin(dates_when_predicting)]
    temp = hfa_period.groupby(['load_week']+granularity)[
        n.FIELD_KNOWN_ORDERS].sum().reset_index()

    temp = temp[temp['calendar_yearweek'].isin(dates_to_predict)]
    temp.rename(columns={'load_week': 'date_when_predicting'}, inplace=True)
    temp.rename(columns={'calendar_yearweek': 'date_to_predict'}, inplace=True)

    return temp


def feature_sumknown_orders(hfa, granularity, dates_when_predicting, dates_to_predict):
    """

    """
    hfa_period = hfa[hfa['load_week'].isin(dates_when_predicting)]
    temp = hfa_period.groupby(['load_week']+granularity)[
        n.FIELD_KNOWN_ORDERS].sum().reset_index()

    temp = temp[temp['calendar_yearweek'].isin(dates_to_predict)]
    temp.rename(columns={'load_week': 'date_when_predicting'}, inplace=True)
    temp.rename(columns={'calendar_yearweek': 'date_to_predict'}, inplace=True)

    res = pd.DataFrame()

    for w in temp['date_when_predicting'].unique():
        orders_known_atweek = temp[temp['date_when_predicting'] == w].copy()
        orders_known_atweek[n.FIELD_KNOWN_ORDERS] = orders_known_atweek[n.FIELD_KNOWN_ORDERS].fillna(0)

        sumvalues = orders_known_atweek.groupby(['plant', 'customer_planning_group', 'lead_sku'])[n.FIELD_KNOWN_ORDERS].cumsum()
        orders_known_atweek[n.FIELD_KNOWN_ORDERS] = sumvalues.values
        orders_known_atweek.rename(columns={n.FIELD_KNOWN_ORDERS: 'sumknown_orders'}, inplace=True)
        orders_known_atweek['date_when_predicting'] = w
        res = pd.concat([res, orders_known_atweek])

    return res
