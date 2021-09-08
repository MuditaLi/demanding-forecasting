""" Sell-out invenorty features - Tested impact on accuracy - No """
import numpy as np
import pandas as pd
import preprocessor.names as n
from functools import partial
from util import misc


def add_ratio_cpg(sales_cpg, cutoff=201801):
    """
    Since for France, the sell-in data scope is bigger than the sell-out one, we need to scale back up sell-out
    sales using a scale corrective ratio. It is the only way to get the consistent sales over time that we need
    to estimate inventory.

    Args:
        cutoff (int): latest data considered to compute the scale corrective ratio.

    Returns:
        Sales for a given CPG (sell-in & sell-out) including the corrective ratio
    """
    # Could try with 25% smallest values instead
    # sales_cpg = sales_cpg[sales_cpg[n.FIELD_YEARMONTH] < 201801]
    pd.options.mode.use_inf_as_na = True
    ratio = (sales_cpg.groupby('group_id')[n.FIELD_LABEL].sum() /
             sales_cpg.groupby('group_id')['total_shipments_volume'].sum()) / 1.1

    ratio = (sales_cpg.groupby(['group_id'])[n.FIELD_LABEL].rolling(window=12).sum().values /
             sales_cpg.groupby(['group_id'])['total_shipments_volume'].rolling(window=12).sum().values)

    sales_cpg['ratio'] = ratio
    for x in range(1, 12):
        sales_cpg['ratio'] = sales_cpg['ratio'].fillna(sales_cpg['ratio'].shift(-x))

    # Ratio approach is misleading when looking at NPD or occasional SKU (best not to correct volumes)
    # Find NPDs
    new_products = sales_cpg[sales_cpg['total_shipments_volume'] != 0].groupby('group_id')[n.FIELD_YEARMONTH].min()
    new_products = new_products[new_products >= cutoff].index
    sales_cpg['ratio'] = sales_cpg['ratio'].where(~sales_cpg['group_id'].isin(new_products), 1)

    sales_cpg['ratio'] = sales_cpg['ratio'].where(sales_cpg['ratio'].abs() < 20, 1)
    return sales_cpg


def adjust_sell_out_volumes(sales_cpg):
    """ Combine sales and the corrective ratio """
    sales_cpg = add_ratio_cpg(sales_cpg)
    sales_cpg = sales_cpg[sales_cpg['ratio'].notnull()]
    sales_cpg['sell_out_adjusted'] = sales_cpg[n.FIELD_LABEL] / sales_cpg['ratio']

    return sales_cpg


def compute_cumulative_volumes(sales_cpg, rolling: bool=False):
    """
    Compute cumulative sell-in & sell-out volumes.

    Args:
        rolling (bool): Whenever True, compute the inventory using the difference between cumulative volumes on a
        12-month rolling window only. Otherwise, we compare cumulative volumes from the earliest point available.

    """

    # Estimated inventory base on 12-month rolling window comparison of cumulative sell-in & sell-out volumes
    if rolling:
        sales_cpg['cumulative_volume_out'] = (sales_cpg.groupby(['group_id'])['sell_out_adjusted']
                                              .rolling(window=12, min_periods=1).sum().values)
        sales_cpg['cumulative_volume_in'] = (sales_cpg.groupby(['group_id'])['total_shipments_volume']
                                             .rolling(window=12, min_periods=1).sum().values)
        return sales_cpg

    sales_cpg['cumulative_volume_out'] = sales_cpg.groupby(['group_id'])['sell_out_adjusted'].cumsum()
    sales_cpg['cumulative_volume_in'] = sales_cpg.groupby(['group_id'])['total_shipments_volume'].cumsum()

    return sales_cpg


def adjust_stocks_to_remove_negative_values(sales_cpg, cutoff=201801):
    """
    Since we do not know the starting inventory, we can end up with negative estimated inventories from the
    difference of cumulative sales volumes. Therefore, we adjust the estimated stock levels to scale them back up to
    positive values, hence ensuring consistency between inventory levels across units and customers.

    0 = lowest ever-seen stock level with that approach.

    Args:
        cutoff (int): atest data considered to compute the scale corrective ratio and the minimum inventory

    Returns:
        Corrective volume needed to bring back inventory levels to positive values.

    """

    # Could try minimum over rolling 12 months period instead
    minimum = dict(
        sales_cpg[sales_cpg[n.FIELD_YEARMONTH] < cutoff].groupby('group_id')['estimated_inventory_adjusted'].min())
    return (sales_cpg['group_id'].map(minimum) * -1).clip(0)


def compute_rolling_inventory(sales_cpg, shift: int=0):
    """
    Estimate inventory from (corrected) sales volumes.

    Args:
        shift (int): Whenever considering rolling weekly sales, there could be a need to add a lag between sell-in
        and sell-out to match peaks.
    """
    sales_cpg['estimated_inventory_adjusted'] = (sales_cpg['cumulative_volume_in']
                                                 - sales_cpg['cumulative_volume_out'].shift(shift))
    if shift:
        sales_cpg['estimated_inventory_adjusted'] = sales_cpg['estimated_inventory_adjusted'].where(
            sales_cpg['group_id'] == sales_cpg['group_id'].shift(shift), np.nan)

    sales_cpg['estimated_inventory_adjusted'] += adjust_stocks_to_remove_negative_values(sales_cpg)
    return sales_cpg


def estimate_sell_out_inventory_cpg(cpg_code: str, sell_in: pd.DataFrame, sell_out: pd.DataFrame, group_ids: list,
                                    shift: int=0):

    sell_in_cpg = sell_in[(sell_in[n.FIELD_CUSTOMER_GROUP] == cpg_code) & (sell_in[n.FIELD_YEARMONTH] >= 201601)]
    sell_out_cpg = sell_out[sell_out[n.FIELD_CUSTOMER_GROUP] == cpg_code]

    # Assign group ids to group lead_sku and EANs that have common matches
    sell_out_cpg['group_id'] = sell_out_cpg[n.FIELD_SKU_EAN].map(group_ids[0])
    sell_in_cpg['group_id'] = sell_in_cpg[n.FIELD_LEAD_SKU_ID].map(group_ids[1])
    sell_in_cpg = sell_in_cpg.groupby(['group_id', n.FIELD_YEARMONTH])['total_shipments_volume'].sum().reset_index()
    sell_out_cpg = sell_out_cpg.groupby(['group_id', n.FIELD_YEARMONTH])[n.FIELD_LABEL].sum().reset_index().fillna(0)

    # Combine sell-in & sell-out
    sales_cpg = sell_in_cpg.merge(sell_out_cpg, on=['group_id', n.FIELD_YEARMONTH], how='left').fillna(0)

    # Compensate for the fact that sell-in scope > sell-out scope (few or no convenient stores in sell-out)
    sales_cpg = adjust_sell_out_volumes(sales_cpg)
    sales_cpg = compute_cumulative_volumes(sales_cpg)
    sales_cpg = compute_rolling_inventory(sales_cpg)

    # Merge with full dataframe
    df_cpg = pd.DataFrame(list(zip(group_ids[1].keys(), group_ids[1].values(), [cpg_code] * len(group_ids[1]))),
                          columns=[n.FIELD_LEAD_SKU_ID, 'group_id', n.FIELD_CUSTOMER_GROUP])

    df_cpg = (sales_cpg[['group_id', n.FIELD_YEARMONTH, 'estimated_inventory_adjusted']]
              .merge(df_cpg, on='group_id', how='left')
              )
    return df_cpg


def build_sell_out_inventory_features(group_ids: list, sell_in_fc_acc: pd.DataFrame, sell_out: pd.DataFrame):

    # TODO - CAREFUL cutoff=201801 parameter was hard-coded to test feature. Not implemented in production
    # TODO - since feature did not improve the accuracy.
    inventories = list()
    for cpg in sell_out[n.FIELD_CUSTOMER_GROUP].unique():
        inventories.append(estimate_sell_out_inventory_cpg(cpg, sell_in_fc_acc, sell_out, group_ids, 0))

    inventories = (pd.concat(inventories, axis=0, sort=False)
                   .rename(columns={n.FIELD_YEARMONTH: n.FIELD_YEARMONTH + '_inventory'})
                   )

    df_features = (sell_in_fc_acc
                   .sort_values(n.FIELD_YEARMONTH, ascending=False)
                   .drop_duplicates([n.FIELD_LEAD_SKU_ID, n.FIELD_YEARWEEK, n.FIELD_CUSTOMER_GROUP])
                   .filter([n.FIELD_LEAD_SKU_ID, n.FIELD_YEARWEEK, n.FIELD_CUSTOMER_GROUP, n.FIELD_YEARMONTH])
                   )

    df_features = df_features.merge(inventories, on=[n.FIELD_CUSTOMER_GROUP, n.FIELD_LEAD_SKU_ID], how='inner')
    df_features = df_features[df_features[n.FIELD_YEARMONTH] > df_features[n.FIELD_YEARMONTH + '_inventory']]
    df_features['delta'] = df_features[[n.FIELD_YEARMONTH + '_inventory', n.FIELD_YEARMONTH]].apply(
        lambda x: partial(misc.diff_period, is_weekly=False)(*x), axis=1)

    # Past 3-months of inventories
    df_features = (df_features[df_features['delta'].isin([1, 2, 3])])

    return (pd.pivot_table(df_features, index=[n.FIELD_LEAD_SKU_ID, n.FIELD_CUSTOMER_GROUP, n.FIELD_YEARWEEK],
                           values='estimated_inventory_adjusted', columns=['delta'], aggfunc=np.sum).reset_index()
            .rename(columns={n.FIELD_YEARWEEK: 'date_to_predict'})
            )

