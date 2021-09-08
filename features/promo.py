import pandas as pd
import util.misc as misc
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from features.pipeline_helpers import TypeSelector, ColumnSelector, clean_ohe_cols
import numpy as np


def features_promos(hfa, granularity, dates_to_predict):
    """
    """
    temp = hfa.groupby(granularity)[
        'promotion_uplift'].sum().reset_index()

    temp = temp[temp['calendar_yearweek'].isin(dates_to_predict)]
    temp.rename(columns={'calendar_yearweek': 'date_to_predict'}, inplace=True)
    # temp[temp.promotion_uplift > 0] = 1
    # temp[temp.promotion_uplift < 0] = -1

    return temp


def create_rawpromo_table(futurmaster):
    """
    :param futurmaster: the table containing futurmaster data
    :return: the table containing the column we want to keep to build features
    """
    res_table = futurmaster.copy()
    res_table = res_table[~res_table.lead_sku.isnull()]
    res_table = res_table[res_table.statut_op.isin(['CLOTURE', 'CONFIRME'])]
    res_table.lead_sku = res_table.lead_sku.astype(int)
    res_table['year_week'] = pd.to_datetime(res_table['date_de_debut_op']).apply(lambda x: x.year).map(str) \
                             + res_table['semaine'].map(str).str.rjust(2, '0')
    res_table['year_week'] = res_table['year_week'].astype(int)

    # keep only main mechanics
    res_table.loc[~res_table.mecanique.isin(
        ['Aucune', 'RI %', 'LV 2eme a 50%', 'LV 3 pour 2', 'Ticket/Cagnottage %']), 'mecanique'] = 'autre_meca'
    res_table.loc[~res_table['type_d\'offre'].isin(
        ['Prospectus', 'Saisonnier', 'Offre Complementaire']), 'type_d\'offre'] = 'autre_type'

    return res_table[['year_week', 'mecanique', 'type_d\'offre', 'natreg', 'lead_sku', 'customer_planning_group',
                      'valeur_mecanique', 'prev_cm_hl']]


def make_prom_pipeline():
    preprocess_pipeline = make_pipeline(
        FeatureUnion(transformer_list=[
            ("categorical_features", make_pipeline(
                TypeSelector("category"),
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(sparse=False, )
            ))
        ])
    )
    return preprocess_pipeline


def get_prom_columns(plant_cpg_pipeline):

    cat_cols = plant_cpg_pipeline.named_steps["featureunion"] \
        .transformer_list[0][1] \
        .named_steps["typeselector"] \
        .get_feature_names()

    ohe_cols = plant_cpg_pipeline.named_steps["featureunion"] \
        .transformer_list[0][1] \
        .named_steps["onehotencoder"] \
        .get_feature_names()
    ohe_cols = clean_ohe_cols(ohe_cols, cat_cols)

    all_cols = ohe_cols
    return all_cols


def dates_promos_raw(futurmaster):
    """
    """
    test_raw = create_rawpromo_table(futurmaster)
    test_raw_cat = misc.convert_categorical(test_raw[['mecanique', 'type_d\'offre', 'natreg']])
    promraw = make_prom_pipeline()
    promraw.fit(test_raw_cat)

    features = pd.DataFrame(data=promraw.transform(test_raw_cat),
                            columns=get_prom_columns(promraw))
    a = pd.concat([test_raw.reset_index(drop=True)[['year_week', 'lead_sku', 'customer_planning_group', 'prev_cm_hl',
                                                    'valeur_mecanique']], features.reset_index(drop=True)], axis=1)

    return a


def features_promos_raw(futurmaster, dates_to_predict, dist=5):
    """
    """
    res = pd.DataFrame()
    temp = dates_promos_raw(futurmaster)
    temp2 = temp.drop_duplicates(subset=['lead_sku', 'customer_planning_group', 'year_week'], keep='first')
    tmp = temp2.copy()
    for i in dates_to_predict:
        tmp['date_to_predict'] = i
        tmp['delta_prom'] = tmp.year_week.apply(lambda x: misc.delta_between_periods(x, i))
        temp_right_delta = tmp[np.abs(tmp['delta_prom']) <= dist]
        temp_right_delta = temp_right_delta.drop_duplicates(subset=['lead_sku', 'customer_planning_group', 'year_week'],
                                                            keep='first')
        res = pd.concat([res, temp_right_delta])
    return res.drop(['year_week'], axis=1).drop_duplicates(subset=['lead_sku', 'customer_planning_group',

                                                                   'date_to_predict'], keep='first')


def features_promos_cannib(fc_acc, skus_clean, futurmaster, dates_to_predict, dist=3):
    """
    :param fc_acc:
    :param skus_clean:
    :param futurmaster:
    :param dates_to_predict:
    :param dist: range of weeks for which we want to look around to check if another promo is happening
    :return: features indicating cannibalization : 1 if another promo is happening for the same sub-brand, brand or pack
    type for a given CPG and SKU
    """
    sku_cpg = fc_acc.groupby(['lead_sku', 'customer_planning_group'])['plant'].count().reset_index()[
        ['lead_sku', 'customer_planning_group']]
    res = pd.DataFrame()
    temp = dates_promos_raw(futurmaster)[['year_week', 'lead_sku', 'customer_planning_group']]
    tempsku = skus_clean[['lead_sku', 'brand_name', 'sub_brand_name', 'unit_per_pack']]
    tempsku = tempsku[~tempsku.lead_sku.isnull()].drop_duplicates(subset=['lead_sku', 'brand_name', 'sub_brand_name'],
                                                                  keep='first')
    temp = pd.merge(temp, tempsku, how='left', on='lead_sku')
    sku_cpg = pd.merge(sku_cpg, tempsku, how='left', on='lead_sku')
    sku_cpg = sku_cpg.drop_duplicates(subset=['lead_sku', 'customer_planning_group'], keep='first')
    for i in dates_to_predict:
        temp['delta_prom'] = temp.year_week.apply(lambda x: misc.delta_between_periods(x, i))
        temp_right_delta = temp[np.abs(temp['delta_prom']) <= dist]
        to_concat = sku_cpg.copy()
        to_concat['date_to_predict'] = i
        to_concat['promo_in_brand'] = 0
        to_concat['promo_in_subbrand'] = 0
        to_concat['promo_in_unitperpack'] = 0

        condition = (to_concat['customer_planning_group'].isin(temp_right_delta['customer_planning_group'].values)) & (
            to_concat['brand_name'].isin(temp_right_delta['brand_name'].values))
        to_concat.loc[condition, 'promo_in_brand'] = 1

        condition = (to_concat['customer_planning_group'].isin(temp_right_delta['customer_planning_group'].values)) & (
            to_concat['sub_brand_name'].isin(temp_right_delta['sub_brand_name'].values))
        to_concat.loc[condition, 'promo_in_subbrand'] = 1

        condition = (to_concat['customer_planning_group'].isin(temp_right_delta['customer_planning_group'].values)) & (
            to_concat['unit_per_pack'].isin(temp_right_delta['unit_per_pack'].values))
        to_concat.loc[condition, 'promo_in_unitperpack'] = 1

        res = pd.concat([res, to_concat])
    return res[['lead_sku', 'customer_planning_group', 'date_to_predict', 'promo_in_brand',
                'promo_in_subbrand', 'promo_in_unitperpack']]