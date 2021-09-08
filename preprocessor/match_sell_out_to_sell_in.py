import os
import pickle
import pandas as pd
import preprocessor.names as n
from configs.configs import Configs
from preprocessor.data_loader import DataLoader
import util.input_output as io
from util.paths import PATH_DATA_FORMATTED, PATH_MODELS


def match_consumer_units_to_lead_skus(sell_out: pd.DataFrame, sell_in: pd.DataFrame, data_loader: DataLoader):
    """
    Method to get all EANs to consider in sell-out data for a given CPG and lead SKU id
    Args:
        sell_out (pd.DataFrame): sell-out data
        sell_in (pd.DataFrame): sell-in data
        data_loader (DataLoader): data_loader object
    """

    if not sell_out.columns.contains(n.FIELD_CUSTOMER_GROUP):
        sell_out = data_loader.add_cpg_to_sell_out_data(sell_out)

    sell_out = sell_out.filter([n.FIELD_CUSTOMER_GROUP, n.FIELD_SKU_EAN]).drop_duplicates()
    sell_in = (sell_in[sell_in[n.FIELD_CUSTOMER_GROUP].isin(sell_out[n.FIELD_CUSTOMER_GROUP])]
               .filter([n.FIELD_CUSTOMER_GROUP, n.FIELD_LEAD_SKU_ID])
               .drop_duplicates()
               )

    # Matched in master tables
    match_ean_material_id = data_loader.load_match_consumer_unit_ean_to_material_ids()
    match_material_id_lead_sku = data_loader.load_match_lead_sku_to_material_ids()
    match_material_id_lead_sku = match_material_id_lead_sku[
        match_material_id_lead_sku[n.FIELD_LEAD_SKU_ID].isin(sell_in[n.FIELD_LEAD_SKU_ID])
    ]

    # Match lead_sku --> material id --> EAN
    match = sell_in.merge(match_material_id_lead_sku, on=[n.FIELD_LEAD_SKU_ID], how='left')
    match = match.merge(match_ean_material_id, on=[n.FIELD_SKU_ID], how='left')

    # See which ones are in the sell-out
    sell_out['in'] = 1  # add column to identify missing EAN reference (sell-out vs sell-in)
    match = match.merge(sell_out, on=[n.FIELD_CUSTOMER_GROUP, n.FIELD_SKU_EAN], how='left')

    # summary table
    match = (match
             .filter([n.FIELD_CUSTOMER_GROUP, n.FIELD_LEAD_SKU_ID, n.FIELD_SKU_EAN, 'in'])
             .drop_duplicates([n.FIELD_CUSTOMER_GROUP, n.FIELD_LEAD_SKU_ID, n.FIELD_SKU_EAN])
             .sort_values([n.FIELD_CUSTOMER_GROUP, n.FIELD_LEAD_SKU_ID])
             )

    match = match[match[n.FIELD_SKU_EAN].notnull()]
    match[n.FIELD_SKU_EAN] = match[n.FIELD_SKU_EAN].astype(int)
    return match.drop(columns=['in'])


def prepare_sell_out_input_sell_in(df_sell_out, match):
    """
    TODO - CAREFUL SINCE WE HAVE MULTIPLE EANs FOR ONE SKU WE DON'T HAVE SUCH A GREAT MATCH
    """

    sell_out = (data_loader.add_cpg_to_sell_out_data(df_sell_out)
                .groupby([n.FIELD_SKU_EAN, n.FIELD_CUSTOMER_GROUP, n.FIELD_YEARMONTH])[n.FIELD_LABEL].sum()
                .reset_index()
                )
    skus_sell_out = sell_out.merge(match, on=[n.FIELD_CUSTOMER_GROUP, n.FIELD_SKU_EAN], how='left')

    return (skus_sell_out
            .groupby([n.FIELD_CUSTOMER_GROUP, n.FIELD_LEAD_SKU_ID, n.FIELD_YEARMONTH])[n.FIELD_LABEL].sum()
            .reset_index()
            .rename(columns={n.FIELD_LABEL: 'total_volume_hl'})
            )


def determine_sets_of_skus_to_group(match: pd.DataFrame):

    results = list()
    combs = list(match[[n.FIELD_SKU_EAN, n.FIELD_LEAD_SKU_ID]].drop_duplicates().to_records(index=False))

    def determine_individual_sets_of_skus_to_group(ref, combinations):
        matches = list(filter(lambda x: (x[0] == ref[0] or x[1] == ref[1]) and x != ref, combinations))
        results = [ref] + matches

        for match in [ref] + matches:
            if match in combinations:
                combinations.remove(match)

        if len(matches) > 1:
            for match in matches:
                res, combinations = determine_individual_sets_of_skus_to_group(match, combinations)
                results += res

        return results, combinations

    while combs:
        res, combs = determine_individual_sets_of_skus_to_group(combs[0], combs)
        results.append(res)

    return list(map(lambda s: (set([x[0] for x in s]), set([x[1] for x in s])), results))


def assign_independent_groups_eans_skus_id(results):
    # Assign corresponding group number
    from functools import reduce

    res = list()
    for index in range(2):
        d = [{item: idx for item in items} for idx, items in enumerate(map(lambda x: x[index], results))]
        res.append(reduce(lambda dc, x: dc.update(x) or dc, d, {}))
    return res


if __name__ == '__main__':

    configs = Configs()
    configs.is_sell_in_model = False
    data_loader = DataLoader(configs)
    data_loader.load_data('sell_out_dts_clean.csv')
    data_loader.load_sell_in_data('fc_acc_week.csv')

    # Exhaustive match between consumer units' EAN and lead sku ids
    match = match_consumer_units_to_lead_skus(data_loader.df_sell_out, data_loader.df_sell_in_fc_acc, data_loader)
    io.write_csv(match, PATH_DATA_FORMATTED, DataLoader.FILE_NAME_SELLOUT_SELLING_MAPPING)

    # Exhaustive independent subgroups of lead SKUs (& EAN) that should be grouped to estimate inventories
    group_ids = assign_independent_groups_eans_skus_id(determine_sets_of_skus_to_group(match))

    with open(os.path.join(PATH_MODELS, 'grouping_ean_skus.pkl'), 'wb') as f:
        pickle.dump(group_ids, f)
