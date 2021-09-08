import pandas as pd
import preprocessor.names as n


def build_seg_features(sell_in_fc_acc, df_sku_seg):
    list_skus = sell_in_fc_acc[n.FIELD_LEAD_SKU_ID].unique()
    list_event_skus = df_sku_seg[df_sku_seg.selection == 'events'][n.FIELD_LEAD_SKU_ID].unique()
    list_promo_skus = df_sku_seg[df_sku_seg.selection == 'promo'][n.FIELD_LEAD_SKU_ID].unique()
    res = pd.DataFrame()
    res[n.FIELD_LEAD_SKU_ID] = list_skus
    res['is_event_sku'] = 0
    res['is_NPD_sku'] = 0
    res['is_promo_sku'] = 0

    # sku started in 2017 and 2018 have a lead sku number starting with 17 18 or 19; temporary hack since a cleanest
    # way to do it is to load NPDs is to read a table that gives creation date of a sku (missing currently)
    cond = ((res[n.FIELD_LEAD_SKU_ID] // 1000 == 18) | (res[n.FIELD_LEAD_SKU_ID] // 1000 == 19) | (
            res[n.FIELD_LEAD_SKU_ID] // 1000 == 17))
    res.loc[cond, 'is_NPD_sku'] = 1
    res.loc[(res[n.FIELD_LEAD_SKU_ID].isin(list_event_skus)), 'is_event_sku'] = 1
    res.loc[(res[n.FIELD_LEAD_SKU_ID].isin(list_promo_skus)), 'is_promo_sku'] = 1
    return res
