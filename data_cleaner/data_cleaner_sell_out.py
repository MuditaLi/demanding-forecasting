import glob
import os
import re
from util import input_output as io
from util.paths import PATH_DATA_RAW, PATH_DATA_FORMATTED
import preprocessor.names as n
import pandas as pd
import numpy as np
from data_cleaner.data_cleaner import clean_col_names


# Other import functions
def import_carrefour_dts(file):
    year_month = re.search(".*/PPV_(\d{6})\d{2}_.*", file)[1]
    dts = io.read_excel(file, sheetname="Sheet 1")
    dts["year_month"] = year_month
    dts["customer"] = "Carrefour"
    dts = clean_col_names(dts)
    dts = dts.rename(columns={"magasins": "store", "libelle": "sku_description",
                              "qts_uvc_ou_uct": "qty_so", "ca_ttc": "revenue_so"})
    dts = dts[["customer", "store", "year_month", "ean", "sku_description", "revenue_so", "qty_so"]]
    print(f'Carrefour - {year_month} completed')
    return dts


def import_intermarche_dts(file):
    year_month = re.search("08021_075_R0200V01XLS_(\d{4}_\d{2}).xlsx", file)[1].replace("_", "")
    dts = io.read_excel(file, sheetname="Liste 0 EAN")

    dts["year_month"] = year_month
    dts["customer"] = "Intermarche"
    dts = clean_col_names(dts)
    dts = dts.rename(columns={"ville": "store", "ca_mois": "revenue_so", "qté_mois": "qty_so"})
    dts = dts[["customer", "store", "year_month", "ean", "revenue_so", "qty_so"]]

    print(f'Intermarché - {year_month} completed')
    return dts


def import_carrefour_mkt_dts(file):
    year_month = re.search("PPV_(\d{6})\d{2}_KRONENBOURG.*.xlsx", file)[1].replace("_", "")
    dts = io.read_excel(file, sheetname="Sheet 1")
    dts["year_month"] = year_month
    dts["customer"] = "Carrefour Market"
    dts = clean_col_names(dts)
    dts = dts.rename(columns={"magasins": "store", "libellé": "sku_description",
                              "qts_uvc_ou_uct": "qty_so", "ca_ttc": "revenue_so"})
    dts = dts[["customer", "store", "year_month", "ean", "sku_description", "revenue_so", "qty_so"]]
    print(f'Carrefour Market - {year_month} completed')
    return dts


def import_franprix_dts(file):
    dts = io.read_excel(file, sheetname="Feuil1")
    dts["customer"] = "Franprix"
    dts = clean_col_names(dts)

    dts["year_month"] = dts["annee"] * 100 + dts["mois"]

    dts = dts.rename(columns={"nom_magasin": "store", "lprdl": "sku_description",
                              "uvc": "qty_so", "cean": "ean", "adresse1": "address",
                              "codepostal": "zip_code", "ville": "city"})
    dts = dts[["customer", "store", "year_month", "ean", "sku_description", "qty_so",
               "address", "zip_code", "city"]]

    print(f'Franprix completed')

    return dts


def import_geant_casino_dts(file):
    year_month = re.search("/Base Y & Y-1/(\d{6})SV_.*.xlsx", file)[1]
    dts = io.read_excel(file, sheetname="1")
    dts["year_month"] = year_month
    dts["customer"] = "Geant - Casino"
    dts = clean_col_names(dts)
    dts = dts.rename(columns={"libelle_magasin": "store", "libelle_produit": "sku_description",
                              "quantite_vendue": "qty_so"})
    dts = dts[["customer", "store", "year_month", "ean", "sku_description", "qty_so"]]

    print(f"Géant Casino - {year_month} completed")

    return dts


def import_leaderprice_dts(file):
    dts = io.read_excel(file, sheetname="Feuil2")
    dts["customer"] = "Leaderprice"
    dts = clean_col_names(dts)

    dts["address"] = np.where(dts["adresse2"].isnull(), dts["adresse1"], dts["adresse1"] + " " + dts["adresse2"])

    dts["year_month"] = dts["annee"] * 100 + dts["mois"]

    dts = dts.rename(columns={"nom_magasin": "store", "cean": "ean", "lprdl": "sku_description",
                              "uvc": "qty_so", "codepostal": "zip_code", "ville": "city"})
    dts = dts[["customer", "store", "year_month", "ean", "sku_description", "qty_so",
               "address", "zip_code", "city"]]
    print("Leaderprice - Completed")
    return dts


def build_dts_file():
    dts_files = [(PATH_DATA_RAW + "/Datasharing/DTS Carrefour/Base Y & Y-1/*", import_carrefour_dts),
                 (PATH_DATA_RAW + "/Datasharing/Datasharing Inter/Base magasin Y & Y-1/*", import_intermarche_dts),
                 (PATH_DATA_RAW + "/Datasharing/DTS Carrefour Market/Base Y & Y-1/*", import_carrefour_mkt_dts),
                 (PATH_DATA_RAW + "/Datasharing/DTS Franprix/003522 BRASSERIES KRONENBOURG - Janv 2016 à Septembre 2017.xlsx",
                 import_franprix_dts),
                 (PATH_DATA_RAW + "/Datasharing/DTS Geant -Casino/Base Y & Y-1/*", import_geant_casino_dts),
                 (PATH_DATA_RAW + "/Datasharing/DTS Leaderprice/002110 BRASSERIES KRONENBOURG - Janv 2016 à Septembre 2017.xlsx", import_leaderprice_dts)
                 ]

    files = [io_func(file) for file_paths, io_func in dts_files for file in glob.glob(file_paths)]

    # TODO 2015 available for some customers using the Y-1 column on 2016 files
    dts = pd.concat(files)

    io.write_csv(dts, PATH_DATA_FORMATTED, "sell_out_dts_clean.csv")


def clean_ean_volume_conversion_data(sell_out_data=None):
    if sell_out_data is None:
        file_name = 'sell_out_dts_clean.csv'
        sell_out_data = (io.read_csv(PATH_DATA_FORMATTED, file_name, usecols=[n.FIELD_SKU_EAN]))

    # Master data for sell-out - reference for conversion of sales volume in units sold to HL.
    file_name = 'NEW Material data.xlsx'
    sheet_name = 'Material data'
    new_skus = (io.read_excel(PATH_DATA_RAW, file_name, sheetname=sheet_name)
                .rename(columns={'EAN Code': n.FIELD_SKU_EAN, 'Product': 'sku_description'}))

    new_skus['secondary_packaging_size_units'] = new_skus['Primary Packaging Size']
    new_skus['primary_packaging_size_cl'] = new_skus['Secondary Packaging Size'] * 100
    new_skus = new_skus[[n.FIELD_SKU_EAN, 'Single item ean', 'sku_description',
                         'primary_packaging_size_cl', 'secondary_packaging_size_units']]

    ean_matched = set(sell_out_data[n.FIELD_SKU_EAN]).intersection(
        set(new_skus[n.FIELD_SKU_EAN]).union(new_skus['Single item ean']))

    nb_rows = sell_out_data.shape[0]
    print('Coverage sell-out data %s' %
          str(round(sell_out_data[sell_out_data[n.FIELD_SKU_EAN].isin(ean_matched)].shape[0] / nb_rows, 3)))

    # Some items are sold as individual items - let's add them back in main Material ID table with the right conversion
    missing_skus = set(sell_out_data[n.FIELD_SKU_EAN]).difference(new_skus[n.FIELD_SKU_EAN])
    single_item_skus = set(missing_skus).intersection(new_skus['Single item ean'])
    tmp = (new_skus[new_skus['Single item ean'].isin(single_item_skus)].copy()
           .filter(list(new_skus.columns.difference([n.FIELD_SKU_EAN]))))
    tmp[n.FIELD_SKU_EAN] = tmp['Single item ean'].astype(int)
    tmp['secondary_packaging_size_units'] = 1

    new_skus = pd.concat([new_skus, tmp], axis=0, sort=False)

    new_skus['convertion_volume_l'] = (new_skus['secondary_packaging_size_units'] *
                                       new_skus['primary_packaging_size_cl'] / 100)

    # TODO - A couple EAN have inconsistent volume conversions
    new_skus = (new_skus
                .sort_values('convertion_volume_l', ascending=False)
                .drop_duplicates(n.FIELD_SKU_EAN))

    io.write_csv(new_skus, PATH_DATA_FORMATTED, 'ean_master_data_clean.csv')

    print('Missing eans: ', missing_skus.difference(new_skus[n.FIELD_SKU_EAN]))


def clean_ean_master_data():
    # Material master data cleaning for sell-out
    file_name_material_master_data_raw = 'materialMD.csv'
    file_name_ean_material_id_mapping = 'sellout_mapping.csv'

    data = (pd.read_csv(os.path.join(PATH_DATA_RAW, file_name_material_master_data_raw), sep=';', encoding='latin-1',
                        na_values=['#', 'Not assigned', 'Not relevant'])
            .rename(columns={'Lead SKU': n.FIELD_LEAD_SKU_ID, 'Material ID': n.FIELD_SKU_ID}))
    data['is_lead_sku'] = data[n.FIELD_SKU_ID].isin(data[n.FIELD_LEAD_SKU_ID])

    ean_sku_match = (io.read_csv(PATH_DATA_FORMATTED, file_name_ean_material_id_mapping)
                     .rename(columns={'Sell_Out_EAN': n.FIELD_SKU_EAN, 'MATNR': n.FIELD_SKU_ID})
                     .filter([n.FIELD_SKU_EAN, n.FIELD_SKU_ID]))

    data = data[data[n.FIELD_SKU_ID].isin(ean_sku_match[n.FIELD_SKU_ID])]
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    relevant_columns = [
        n.FIELD_SKU_ID,
        'pack_size_name',
        'unit_per_pack',
        'material_sub-group',
        'container_type',
        'global_bev_cat_name',
        'brand_name',
        'alcohol_percentage',
        'is_lead_sku'
    ]
    pattern = re.compile(r'(CAN|BOT|KEG)', re.MULTILINE | re.IGNORECASE)
    data['pack_size_name'] = data['pack_size_name'].str.replace(',', '.').astype(float)
    data['container_type'] = data['material_name'].str.extract(pattern, expand=True)
    data = data[relevant_columns].drop_duplicates()

    cleaned_information = (data
                           .drop(columns=['alcohol_percentage'])
                           .groupby(n.FIELD_SKU_ID).agg(pd.Series.mode).reset_index())

    # Alcohol percentage information
    # Take most frequent
    tmp = (data[(data['alcohol_percentage'].notnull()) & (data['alcohol_percentage'] != 0)]
           .groupby(n.FIELD_SKU_ID)['alcohol_percentage']
           .mean()
           .reset_index())

    cleaned_information = cleaned_information.merge(tmp, on=n.FIELD_SKU_ID, how='left')

    # Add 0 degrees beer
    tmp = (data[(data['alcohol_percentage'] == 0) & (~data[n.FIELD_SKU_ID].isin(tmp[n.FIELD_SKU_ID]))]
           .filter([n.FIELD_SKU_ID, 'alcohol_percentage'])
           .drop_duplicates(n.FIELD_SKU_ID))

    cleaned_information = cleaned_information.merge(tmp, on=n.FIELD_SKU_ID, how='left', suffixes=['', '_x'])
    cleaned_information['alcohol_percentage'] = cleaned_information['alcohol_percentage'].fillna(
        cleaned_information['alcohol_percentage_x'])

    # Cleaned_information at material_id level
    cleaned_information = cleaned_information.filter(relevant_columns).astype(str)
    cleaned_information = cleaned_information.replace('[]', np.nan)

    cleaning = [
        (n.FIELD_SKU_ID, int),
        ('pack_size_name', float),
        ('unit_per_pack', int),
        ('alcohol_percentage', float),
        ('is_lead_sku', bool)
    ]

    for x, y in cleaning:
        cleaned_information[x] = cleaned_information[x].astype(y)

    # Features EAN level
    aggregation_strategy = {
        'alcohol_percentage': np.mean,
        'unit_per_pack': np.max,
        'pack_size_name': np.mean,
    }
    tmp = (ean_sku_match
           .merge(cleaned_information, on=n.FIELD_SKU_ID)
           .drop(columns=[n.FIELD_SKU_ID])
           .drop_duplicates())

    ean_product_characteristics = (tmp
                                   .filter(tmp.columns.difference(list(aggregation_strategy)))
                                   .sort_values('material_sub-group')
                                   .drop_duplicates(n.FIELD_SKU_EAN)
                                   .merge(tmp.groupby(n.FIELD_SKU_EAN).aggregate(aggregation_strategy).reset_index(),
                                          on=n.FIELD_SKU_EAN, how='left')
                                   )

    relevant_columns = [
        n.FIELD_SKU_EAN,
        'brand_name',
        'brand_name',
        'container_type',
        'global_bev_cat_name',
        # 'is_lead_sku',
        'material_sub-group',
        'alcohol_percentage',
        'unit_per_pack',
        'pack_size_name'
    ]
    ean_product_characteristics = (ean_product_characteristics
                                   .filter(relevant_columns)
                                   .rename(columns={'material_sub-group': 'material_sub_group'}))
    ean_product_characteristics.to_csv(os.path.join(PATH_DATA_FORMATTED, 'ean_product_characteristics.csv'))


if __name__ == "__main__":
    build_dts_file()
    clean_ean_volume_conversion_data()
    clean_ean_master_data()
