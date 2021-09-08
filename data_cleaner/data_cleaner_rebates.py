import os
import numpy as np
import pandas as pd
from difflib import get_close_matches

import util.input_output as io
import preprocessor.names as n
from util.paths import PATH_DATA_RAW, PATH_DATA_FORMATTED


# Static files
file_name_sales_analyzer_ship_to_mapping = 'SALES_ANALYZER_2016_2018.csv'
file_name_customer_table_clean = 'cpg_sold_to_clean.csv'
file_name_material_lead_sku_mapping = 'materialMD.csv'


def load_material_lead_sku_mapping_table():
    data = (io.read_csv(PATH_DATA_RAW, file_name_material_lead_sku_mapping, delimiter=';',
                        usecols=['Material ID', 'Lead SKU'])
            .drop_duplicates()
            .rename(columns={'Material ID': n.FIELD_SKU_ID, 'Lead SKU': n.FIELD_LEAD_SKU_ID})
            )
    data = data[data[n.FIELD_LEAD_SKU_ID].notnull()]

    for col in [n.FIELD_SKU_ID, n.FIELD_LEAD_SKU_ID]:
        data[col] = data[col].astype(int)

    return data


def format_as_week(dates):
    # Issue with 12-31 dates --> 1st week of next year
    dates = pd.to_datetime(dates, format='%Y%m%d')

    # Match Carlsberg's week number format (issues with first & last week of a pandas year where year is misleading)
    correction = np.where((dates.dt.month == 12) & (dates.dt.week == 1), 1, 0)
    correction = np.where((dates.dt.month == 1) & (dates.dt.week >= 52), -1, correction)
    year = dates.dt.year + correction
    week = dates.dt.week.astype(str).str.zfill(2)
    return (year.astype(str) + week).astype(int)


def clean_header(header):
    if isinstance(header, list):
        header = pd.Series(header)

    header = (header
              .str.replace('|'.join(['\t', '/', '-', '\(', '\)', '"']), '_')
              .str.strip('_')
              .str.lower()
              .str.split()
              .str.join('_')
              )
    return header


def clean_ship_to_cpg_mapping_table(ship_tos_to_match):

    ship_to_mapping = pd.read_csv(os.path.join(PATH_DATA_RAW, 'SalesShipTo', file_name_sales_analyzer_ship_to_mapping),
                                  na_values=['#', '000'])
    ship_to_mapping.rename(columns={ship_to_mapping.columns[-2]: n.FIELD_CUSTOMER}, inplace=True)

    acc = '|'.join(['\n', '\-'])
    ship_to_mapping.columns = map(lambda x: '_'.join(x.replace(acc, ' ').lower().split()), ship_to_mapping.columns)
    ship_to_mapping = ship_to_mapping[ship_to_mapping['customer_planning_group'] != 'Result']
    ship_to_mapping = ship_to_mapping[ship_to_mapping['ship-to'].str.lower().str.extract(r'([a-z])')[0].isnull()]
    ship_to_mapping['ship-to'] = ship_to_mapping['ship-to'].astype(int)
    ship_to_mapping['rebate_recipient'] = ship_to_mapping['ship-to']

    return add_closest_match_for_missing_customer_cpg(ship_to_mapping, ship_tos_to_match)


def add_closest_match_for_missing_customer_cpg(ship_to_cpg_mapping, ship_tos_to_match):

    # Load cleaned customer table
    customer_table = io.read_csv(PATH_DATA_FORMATTED, file_name_customer_table_clean,
                                 usecols=['customer', 'sold_to_party_id', n.FIELD_CUSTOMER_GROUP])

    # Split data to consider only missing cpg assignments
    matched = (ship_to_cpg_mapping
               .groupby(['rebate_recipient', n.FIELD_CUSTOMER_GROUP])['ship-to'].count()
               .reset_index()
               .sort_values('ship-to', ascending=False)
               .drop_duplicates('rebate_recipient')
               .drop(columns=['ship-to'])
               )

    not_matched = set(ship_tos_to_match).difference(matched['rebate_recipient'].unique())
    not_matched = (ship_to_cpg_mapping[ship_to_cpg_mapping['rebate_recipient'].isin(not_matched)]
                   .filter([n.FIELD_CUSTOMER, 'rebate_recipient'])
                   )

    # customer_table[customer_table[n.FIELD_CUSTOMER].str.contains('ATAC ')]
    def find_best_match(target, candidates):
        target = target.replace('PROVERA', '').replace('SUPERMARCHE', '')
        match = get_close_matches(target, candidates)
        target = set(target.split())
        words_in_common = [len(set(target).intersection(m.split())) for m in match]
        return match[words_in_common.index(max(words_in_common))]

    # We can find missing match in the customer table (closest match and assign corresponding CPG)
    not_matched[n.FIELD_CUSTOMER] = not_matched[n.FIELD_CUSTOMER].str.replace(r' -R$', '')
    not_matched[n.FIELD_CUSTOMER] = not_matched[n.FIELD_CUSTOMER].str.replace(r'(F)[0-9]', '')

    match = dict()
    customers = customer_table[n.FIELD_CUSTOMER].str.replace('SUPERMARCHE', '').unique()
    for cust in not_matched[n.FIELD_CUSTOMER]:
        if cust in customer_table[n.FIELD_CUSTOMER].values:
            continue

        match.update({cust: find_best_match(cust, customers)})

    # manual corrections
    match.update({'ATAC SUPERMARCHE': 'AVENANCE ATAC LOGISTIQUE',
                  'PROVERA MATCH': 'SUPERMARCHES MATCH',
                  'PROVERA CORA': 'CORA S1'})

    print('Match for missing cpg assignments')
    print(match)

    not_matched['key'] = not_matched[n.FIELD_CUSTOMER].map(match).fillna(not_matched[n.FIELD_CUSTOMER])
    not_matched = (not_matched.merge(customer_table.rename(columns={n.FIELD_CUSTOMER: 'key'}), on='key', how='left')
                   .filter(['rebate_recipient', n.FIELD_CUSTOMER_GROUP]))

    return pd.concat([matched, not_matched], axis=0, sort=False).drop_duplicates('rebate_recipient')


def clean_raw_rebates_table(file_name, output_file_name):

    header = io.read_csv(PATH_DATA_RAW, 'Rebates', file_name, sep=';', error_bad_lines=False, nrows=1, header=None)
    data = io.read_csv(PATH_DATA_RAW, 'Rebates', file_name, sep=';', error_bad_lines=False, header=None, skiprows=1)
    header = clean_header(list(header.loc[0, header.columns.difference([12])].values) +
                          header.loc[0, 12].strip('\t').split('\t'))
    data.rename(columns=dict(zip([i for i in range(33) if i not in [12, 14, 16]], header)), inplace=True)

    data[n.FIELD_YEARWEEK + '_start'] = format_as_week(data['valid_from_date'])
    data[n.FIELD_YEARWEEK + '_end'] = format_as_week(data['valid_to_date'])

    types = {
        'agreement_type': str,
        'agreement_status': str,
        'customer_cust_hier_dem_sector': int,
        'rebate_recipient': int,
        'accrual_rate': float,
        'accrual_rate_%': float,
        'current_scale_base_value': float,
        'current_scale_base_volume': float,
        'accruals_amount': float
    }
    for col, tpe in types.items():
        data[col] = data[col].astype(tpe)

    data[n.FIELD_REBATE_MECHANISM] = np.where(data['accrual_rate'] == 0, 'value', 'volume')
    data['rebate_rate_unit'] = np.where(data['accrual_rate'] == 0, '%', 'EUR/HL')
    data['rebate_condition_unit'] = np.where(data['accrual_rate'] == 0, 'EUR', 'HL')

    data[n.FIELD_REBATE_TARGET] = data['current_scale_base_value'].where(
        data['accrual_rate'] == 0, data['current_scale_base_volume'])

    data[n.FIELD_REBATE_RATE] = data['accrual_rate_%'].where(
        data['accrual_rate'] == 0, data['accrual_rate'])

    cpg_shipto_mapping = clean_ship_to_cpg_mapping_table(data['rebate_recipient'].unique())
    data = data.merge(cpg_shipto_mapping, on='rebate_recipient', how='left')
    print('%d rows dropped - No CPG match' % data[n.FIELD_CUSTOMER_GROUP].isnull().sum())

    relevant_columns = [
        'agreement_number',
        'agreement_type',
        'agreement_status',
        'valid_from_date',
        'valid_to_date',
        n.FIELD_YEARWEEK + '_start',
        n.FIELD_YEARWEEK + '_end',
        'rebate_recipient',
        'customer_cust_hier_dem_sector',
        n.FIELD_SKU_ID,
        n.FIELD_REBATE_RATE,
        'rebate_rate_unit',
        n.FIELD_REBATE_TARGET,
        'rebate_condition_unit',
        n.FIELD_REBATE_MECHANISM,
        'accruals_amount',
        n.FIELD_CUSTOMER_GROUP
    ]
    data = data.loc[data[n.FIELD_CUSTOMER_GROUP].notnull(), relevant_columns]

    # Add corresponding lead_sku id
    data = data.merge(load_material_lead_sku_mapping_table(), on=n.FIELD_SKU_ID)
    io.write_csv(data, PATH_DATA_FORMATTED, output_file_name)


def clean_inconsistent_rows(file_name, file_name_clean):

    with open(os.path.join(PATH_DATA_RAW, 'Rebates', file_name), 'r') as f:
        rows = f.readlines()

    from collections import Counter
    length_rows = [len(row.split(';')) for row in rows]  # replace(';;', ';')
    standard_size = Counter(length_rows).most_common(1)
    idx = [i for i in range(1, len(rows)) if length_rows[i] != standard_size[0][0]]

    idx_desc = 8
    for i in idx:
        row = rows[i].split(';')
        rows[i] = ';'.join(row[:idx_desc] + [row[idx_desc] + row[idx_desc + 1]] + row[idx_desc + 2:])

    with open(os.path.join(PATH_DATA_RAW, 'Rebates', file_name_clean), 'w') as f:
        f.write('\n'.join(rows))


if __name__ == '__main__':

    clean_inconsistent_rows('20190326_rebates_extract.csv', '20190326_rebates_extract_clean.csv')
    clean_raw_rebates_table('20190326_rebates_extract_clean.csv', 'rebates_clean.csv')
