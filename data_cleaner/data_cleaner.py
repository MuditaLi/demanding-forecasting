import glob
import re
import os
import json
import requests
from util import input_output as io
from util.paths import PATH_DATA_RAW, PATH_DATA_FORMATTED
import preprocessor.names as n
from util import misc
import pandas as pd
import numpy as np
import unidecode


def loader(path):
    return io.read_csv(PATH_DATA_RAW, path, delimiter=";", skiprows=[0, 1, 3])


def format_string_names(obj):
    return obj.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.upper()


def clean_forecastaccuracy_monthbias():
    # Load data
    files = ["ForecastAccuracyBiasMonth2015_2019/ForecastAccuracyMonthBias2015_2016.txt",
             "ForecastAccuracyBiasMonth2015_2019/ForecastAccuracyMonthBias1.txt",
             "ForecastAccuracyBiasMonth2015_2019/ForecastAccuracyMonthBias2.txt"]
    fc_acc_month = pd.concat(list(map(loader, files)))

    # Clean data
    fc_acc_month = clean_col_names(fc_acc_month)

    cols_to_drop = [_ for _ in fc_acc_month.columns if "unit" in _]
    fc_acc_month = fc_acc_month.drop(columns=cols_to_drop)

    fc_acc_month.lead_sku = clean_lead_sku(fc_acc_month.lead_sku)

    # Save clean data
    io.write_csv(fc_acc_month, PATH_DATA_FORMATTED, "fc_acc_bias_month_clean.csv")


def clean_fcacc_month():
    #Load data
    files = ["ForecastAccuracyBiasMonth2015_2019/FCBIAS_MTHLY_2015.csv",
             "ForecastAccuracyBiasMonth2015_2019/FCBIAS_MTHLY_2016.csv",
             "ForecastAccuracyBiasMonth2015_2019/FCBIAS_MTHLY_2017.csv",
             "ForecastAccuracyBiasMonth2015_2019/FCBIAS_MTHLY_2018.csv"]
    list_df = list()
    for f in files:
        list_df.append(io.read_csv(PATH_DATA_RAW, f, delimiter=";"))
    fc_acc_month = pd.concat(list_df)

    # Clean data
    fc_acc_month = clean_col_names(fc_acc_month)

    cols_to_drop = [_ for _ in fc_acc_month.columns if "unit" in _]
    fc_acc_month = fc_acc_month.drop(columns=cols_to_drop)

    fc_acc_month.lead_sku = clean_lead_sku(fc_acc_month.lead_sku)

    # Save clean data
    io.write_csv(fc_acc_month, PATH_DATA_FORMATTED, "fc_acc_month.csv")


def clean_fcacc_week():
    # Load data
    files = ["ForecastAccuracyBias2015_2019/FAB_FR_XTRACT_2015.csv",
             "ForecastAccuracyBias2015_2019/FAB_FR_XTRACT_2016.csv",
             "ForecastAccuracyBias2015_2019/FAB_FR_XTRACT_2017.csv",
             "ForecastAccuracyBias2015_2019/FAB_FR_XTRACT_2018.csv"]
    list_df = list()
    for f in files:
        list_df.append(io.read_csv(PATH_DATA_RAW, f, delimiter=";"))
    fc_acc = pd.concat(list_df)

    # Clean data
    fc_acc = clean_col_names(fc_acc)

    cols_to_drop = [_ for _ in fc_acc.columns if "unit" in _]
    fc_acc = fc_acc.drop(columns=cols_to_drop)

    fc_acc.lead_sku = clean_lead_sku(fc_acc.lead_sku)

    # Save clean data
    io.write_csv(fc_acc, PATH_DATA_FORMATTED, "fc_acc_week.csv")


def clean_hfa():
    # Load data
    files = ["HFA_LW/HFA_DCD_F001_012016_062016.csv",
             "HFA_LW/HFA_DCD_F001_072016_122016.csv",
             "HFA_LW/HFA_DCD_F001_132016_202016.csv",
             "HFA_LW/HFA_DCD_F001_212016_302016.csv",
             "HFA_LW/HFA_DCD_F001_312016_402016.csv",
             "HFA_LW/HFA_DCD_F001_412016_522016.csv",
             "HFA_LW/HFA_DCD_F001_012017_122017.csv",
             "HFA_LW/HFA_DCD_F001_132017_252017.csv",
             "HFA_LW/HFA_DCD_F001_262017_402017.csv",
             "HFA_LW/HFA_DCD_F001_412017_522017.csv",
             "HFA_LW/HFA_DCD_F001_012018_072018.csv",
             "HFA_LW/HFA_DCD_F001_072018_152018.csv",
             "HFA_LW/HFA_DCD_F001_162018_252018.csv",
             "HFA_LW/HFA_DCD_F001_262018_352018.csv",
             "HFA_LW/HFA_DCD_F001_362018_442018.csv",
             "HFA_LW/HFA_DCD_F001_452018_522018.csv"
             ]

    cols = ['Calendar Year/Month', 'Calendar year/week', 'Load Week', 'Customer Planning Group', 'Lead SKU', 'Plant',
            'Total Shipments', 'Open Orders', 'Promotion Uplift', 'Total Sales Volume']

    list_df = list()
    for f in files:
        print(f)
        tem_df = io.read_csv(PATH_DATA_RAW, f)
        tem_df = tem_df[~tem_df['Lead SKU'].isnull()]
        tem_df['Lead SKU'] = tem_df['Lead SKU'].astype(float).astype(int)
        list_df.append(tem_df[cols])
    hfa = pd.concat(list_df)

    # Clean data
    hfa = clean_col_names(hfa)

    cols_to_drop = [_ for _ in hfa.columns if "unit" in _]
    hfa = hfa.drop(columns=cols_to_drop)

    # hfa.lead_sku = clean_lead_sku(hfa.lead_sku)
    hfa.drop_duplicates(inplace=True)

    # Save clean data
    io.write_csv(hfa, PATH_DATA_FORMATTED, "hfa_load_weeks.csv")


def clean_hfa20152019():
    hfa = io.read_csv(PATH_DATA_FORMATTED, 'hfa_clean.csv', delimiter=';')
    # Clean data
    hfa = clean_col_names(hfa)

    cols_to_drop = [_ for _ in hfa.columns if "unit" in _]
    hfa = hfa.drop(columns=cols_to_drop)

    hfa.lead_sku = clean_lead_sku(hfa.lead_sku)

    # Save clean data
    io.write_csv(hfa, PATH_DATA_FORMATTED, "hfa_cleaned.csv")


def clean_col_names(data):
    data.columns = [_.lower().replace(" ", "_").replace("/", "").replace('(', "").replace(")", "").replace(".", "") for _ in data.columns]

    return data


def clean_hist_forecast():
    # Load data
    files = ["HistoricalForecast2015_2019/HistoricalForecast2015_2016.txt",
             "HistoricalForecast2015_2019/HistoricalForecast2017_2019.txt"]
    hist_forecast = pd.concat(list(map(loader, files)))

    # Clean data
    hist_forecast = clean_col_names(hist_forecast)

    cols_to_drop = [_ for _ in hist_forecast.columns if "unit" in _ or "est_" in _] + \
                   ["budget_apo", "apo_calculated_sales_estimate"]
    hist_forecast = hist_forecast.drop(columns=cols_to_drop)
    hist_forecast.lead_sku = clean_lead_sku(hist_forecast.lead_sku)

    # Save clean data
    io.write_csv(hist_forecast, PATH_DATA_FORMATTED, "actual_sales_and_fc_clean.csv")


def clean_lead_sku(lead_sku):
    new_lead_sku = np.where((lead_sku == "#") | (lead_sku.isnull()), -1, lead_sku.str[-5:])
    new_lead_sku = new_lead_sku.astype(int)
    return new_lead_sku


def clean_forecastaccuracy_bias():
    # Load data
    files = ["ForecastAccuracyBias2015_2019/ForecastAccuracyBias2015_2016.txt",
             "ForecastAccuracyBias2015_2019/ForecastAccuracyBias201901.txt",
             "ForecastAccuracyBias2015_2019/ForecastAccuracyBias201902.txt"]
    fc_acc = pd.concat(list(map(loader, files)))

    # Clean data
    fc_acc = clean_col_names(fc_acc)

    cols_to_drop = [_ for _ in fc_acc.columns if "unit" in _]
    fc_acc = fc_acc.drop(columns=cols_to_drop)

    fc_acc.lead_sku = clean_lead_sku(fc_acc.lead_sku)

    # Save clean data
    io.write_csv(fc_acc, PATH_DATA_FORMATTED, "fc_acc_bias_clean.csv")


def clean_activities():
    # Load data
    activities = io.read_csv(PATH_DATA_RAW, "activities.csv")

    # Clean data
    activities = clean_col_names(activities)

    # Save clean data
    io.write_csv(activities, PATH_DATA_FORMATTED, "activities_clean.csv")


def make_int(text):
    return int(float(text.replace(" ", "")))


def make_float(text):
    return float(text.replace("#", ""))


def clean_materials_md():
    materials = io.read_csv(PATH_DATA_RAW, "materialMD.csv", sep=";", converters={"Total Shelf Life": make_int})

    materials = clean_col_names(materials)
    # We need to drop many rows from the material file. The actual index is not clear. It is larger than
    # (material_id, sales_organization_id, plant_id) as there remain duplicates when filtering on plant_id and
    # sales_organization_id equal to F001. When there remain more than 1 row for a given material, we keep the one
    # with the least amount of "Not assigned".
    materials = materials.replace("Not assigned", np.nan)
    materials["pct_na"] = materials.isnull().mean(axis=1)
    materials = materials.loc[(materials.sales_organization_id == "F001") & (materials.plant_id == "F001")] \
        .sort_values("pct_na") \
        .groupby("material_id", as_index=False) \
        .first()

    drop_cols = ["total_rl_time", "safety_stock", "vm_mat_group_id", "vm_mat_group_name",
                 "purchasing_group_id", "purchasing_group_name", "mrp_controller_id", "mixed_mrp",
                 "procurement_type_id", "procurement_type_name", "material_group_id",
                 "material_group_name", "local_brand_group", "grouping_fro_edi", "bwd_cons._per",
                 "material_class.abc", "company_code_id", "company_code_name", "liquid_material_number",
                 "liquid_type", "primary_pack_material_number", "primary_pack_material_name", "closure_type",
                 "neck_label_type", "front_label_type", "back_label_type", "material_sub-group",
                 "material_type_id", "material_type_name", "materialgroup_3_id", "materialgroup_3_name",
                 "materialgroup_4", "materialgroup_4_name", "bwd_cons._per.", "consumption_mode",
                 "fwd_cons.period", "inhseprodtime", "gr_proccessing_time", "planned_deliv.time",
                 "component_scrap_in_percent", "material_group_6_id", "material_group_6_name",
                 "materialgroup_2_id", "materialgroup_2_id.1", "materialgroup_5_id", "materialgroup_5_name",
                 "production_type_id", "production_type_name", "pct_na"]

    materials = materials.drop(columns=drop_cols, errors="ignore")
    materials = materials.rename(columns={"material_id": "material"})

    sku_list = io.read_excel(PATH_DATA_RAW, "SKU", "List SKU.xlsx", sheetname="C1_STD_ANALYSIS_TEMPLATE (1)",
                             skiprows=6)
    sku_list = sku_list.rename(
        columns={"Unnamed: 16": "brand_name", "Unnamed: 18": "subbrand_name", "Unnamed: 71": "Total Shelf Life"})
    selected_sku = sku_list[
        ['Material', 'Material Group', 'Brand', 'Sub Brand', 'subbrand_name']]
    selected_sku = clean_col_names(selected_sku)

    clean_sku = pd.merge(selected_sku, materials, on="material", how="outer")

    io.write_csv(clean_sku, PATH_DATA_FORMATTED, "skus_clean.csv")


def clean_vmi():
    vmi = io.read_excel(PATH_DATA_RAW, "Link Production.xlsx", sheetname="Feuil2", skiprows=3)

    vmi = clean_col_names(vmi)

    id_cols = ["ship_to_name", "produit_code_ean_pal", "produit_code_ean_uc", "produit_code_sku"]
    for col in id_cols:
        vmi[col] = vmi[col].fillna(method="ffill")

    vmi = vmi[vmi.ship_to_name != "Total général"].drop(columns="total_général")

    var_cols = [_ for _ in vmi.columns if _ not in id_cols]
    vmi = vmi.melt(id_cols, var_cols, "date", "stock")

    vmi.date = pd.to_datetime(vmi.date, format="%d%m%Y")

    io.write_csv(vmi, PATH_DATA_FORMATTED, "vmi_clean.csv")


def clean_stock():
    files = ["StockFR/StockFR_2015_2016.txt", "StockFR/StockFR_2017_2019.txt"]
    stock = pd.concat(list(map(loader, files)))

    stock = clean_col_names(stock)
    stock = stock.drop(columns=["total_stock"])
    stock.calendar_day = pd.to_datetime(stock.calendar_day, format="%Y%m%d")

    io.write_csv(stock, PATH_DATA_FORMATTED, "stock_clean.csv")


def clean_iri_sell_out_volume():
    iri_so_volume = io.read_excel(PATH_DATA_RAW, "IRI_data/IRI.xlsx", sheetname="Sheet1", skiprows=3)

    iri_so_volume = iri_so_volume.rename(columns={"Unnamed: 0": "sku"})
    iri_so_volume.columns = [col[7:17] if i >= 1 else col for i, col in enumerate(iri_so_volume.columns)]
    var_cols = iri_so_volume.columns[1:]
    iri_so_volume = iri_so_volume.melt("sku", var_cols, "first_dow", "sell_out_volume")
    iri_so_volume["sku_description"], iri_so_volume["EAN"] = iri_so_volume['sku'].str.split(' - ', 1).str

    # We still need to convert first_dow to calendar_week, but before that we need to make sure all
    # calendar_yearweeks have the same conventions and adapt the misc.get_yearweek() for that.

    # We are also not writing to disk yet, as the current file is only an extract of the cube


def clean_skus_uom():
    sku_uom = io.read_excel(PATH_DATA_RAW, "SKU", "SKU UoM.xlsx", sheetname="C1_STD_ANALYSIS_TEMPLATE", skiprows=5)
    sku_uom = sku_uom.rename(columns={"Unnamed: 1": "material_description", "Unnamed: 3": "Sales_organization_name",
                                      "Denominnator": "Denominator"})
    sku_uom = clean_col_names(sku_uom)

    io.write_csv(sku_uom, PATH_DATA_FORMATTED, "sku_uom_clean.csv")


def clean_futurmaster(country, folder_extract):

    fm2015 = io.read_excel(PATH_DATA_RAW, "FuturMaster", "2019_02_08 Extract FM 2015_Envoi DCD.xls.xlsx", skiprows=15,
                           sheetname="Z_CM_010_2015 War Room CM - Sui", encoding='latin-1')

    fm2016 = io.read_excel(PATH_DATA_RAW, "FuturMaster", "2019_02_01 Extract FM 2016_Envoi DCD.xls.xlsx",  skiprows=15,
                           sheetname="Z_CM_010_2016 War Room CM - Sui", encoding='latin-1')

    fm2017 = io.read_excel(PATH_DATA_RAW, "FuturMaster", "2019_02_01 Extract FM 2017_Envoi DCD.xls.xlsx", skiprows=15,
                           sheetname="Z_CM_010_2017 War Room CM - Sui", encoding='latin-1')

    fm2018 = io.read_excel(PATH_DATA_RAW, "FuturMaster", "2019_03_04 Extract FM 2018_Envoi DCD.xls.xlsx", skiprows=15,
                           sheetname="Z_CM_010_2018 War Room CM - Sui", encoding='latin-1')

    fm2019 = io.read_excel(PATH_DATA_RAW, "FuturMaster", "2019_05_02 Extract FM 2019_Envoi DCD_v1.xlsx", skiprows=16,
                           sheetname="Z_CM_010_2019 War Room CM - Sui", encoding='latin-1')

    fm2015 = clean_col_names(fm2015)
    fm2015.columns = [unidecode.unidecode(x) for x in fm2015.columns]
    fm2015['mecanique'] = fm2015['mecanique'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2015['natreg'] = fm2015['natreg'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2015['regroupement_dv'] = fm2015['regroupement_dv'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2015['op'] = fm2015['op'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2015['type_d\'offre'] = fm2015['type_d\'offre'].astype(str).apply(lambda x: unidecode.unidecode(x))

    fm2018 = clean_col_names(fm2018)
    fm2018.columns = [unidecode.unidecode(x) for x in fm2018.columns]
    fm2018['mecanique'] = fm2018['mecanique'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2018['natreg'] = fm2018['natreg'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2018['regroupement_dv'] = fm2018['regroupement_dv'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2018['op'] = fm2018['op'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2018['type_d\'offre'] = fm2018['type_d\'offre'].astype(str).apply(lambda x: unidecode.unidecode(x))

    fm2017 = clean_col_names(fm2017)
    fm2017.columns = [unidecode.unidecode(x) for x in fm2017.columns]
    fm2017['mecanique'] = fm2017['mecanique'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2017['natreg'] = fm2017['natreg'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2017['regroupement_dv'] = fm2017['regroupement_dv'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2017['op'] = fm2017['op'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2017['type_d\'offre'] = fm2017['type_d\'offre'].astype(str).apply(lambda x: unidecode.unidecode(x))

    fm2016 = clean_col_names(fm2016)
    fm2016.columns = [unidecode.unidecode(x) for x in fm2016.columns]
    fm2016['mecanique'] = fm2016['mecanique'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2016['natreg'] = fm2016['natreg'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2016['regroupement_dv'] = fm2016['regroupement_dv'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2016['op'] = fm2016['op'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2016['type_d\'offre'] = fm2016['type_d\'offre'].astype(str).apply(lambda x: unidecode.unidecode(x))

    fm2019 = clean_col_names(fm2019)
    fm2019.columns = [unidecode.unidecode(x) for x in fm2019.columns]
    fm2019['mecanique'] = fm2019['mecanique'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2019['natreg'] = fm2019['natreg'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2019['regroupement_dv'] = fm2019['regroupement_dv'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2019['op'] = fm2019['op'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2019['type_d\'offre'] = fm2019['type_d\'offre'].astype(str).apply(lambda x: unidecode.unidecode(x))
    fm2019 = fm2019[[col for col in fm2019 if col in fm2018.columns]]

    futurmaster = pd.concat([fm2015, fm2016, fm2017, fm2018, fm2019], sort=False)

    futurmaster.date_de_debut_op = pd.to_datetime(futurmaster.date_de_debut_op, format='%d/%m/%Y')
    futurmaster.date_de_fin_op = pd.to_datetime(futurmaster.date_de_fin_op, format='%d/%m/%Y')
    mat = io.read_csv(PATH_DATA_RAW, 'materialMD.csv', delimiter=';')
    # mat['Material ID'] == pd.to_numeric(mat['Material ID'], errors='coerce')
    mat['Material ID'] = mat['Material ID'].astype(int)
    futurmaster['Material ID'] = futurmaster.sku.apply(lambda x: x[0:5])

    futurmaster['Material ID'] = pd.to_numeric(futurmaster['Material ID'], errors='coerce')

    futurmaster = futurmaster[~futurmaster['Material ID'].isnull()]
    futurmaster['Material ID'] = futurmaster['Material ID'].astype(int)
    futurmaster = pd.merge(futurmaster, mat[['Material ID', 'Lead SKU']].drop_duplicates(),
                           how='left', on='Material ID')
    futurmaster.rename(columns={'Lead SKU': 'lead_sku'}, inplace=True)
    enseigne_cpg = (io.read_csv(PATH_DATA_FORMATTED, "enseigne_cpg.csv", delimiter=';')
                    .filter(['enseigne', 'customer_planning_group']))
    # enseigne_cpg = io.read_csv(PATH_DATA_FORMATTED, "CPG_enseignes.csv", delimiter=';')

    futurmaster = pd.merge(futurmaster, enseigne_cpg, how='left', on='enseigne')
    io.write_csv(futurmaster, PATH_DATA_FORMATTED, country, folder_extract, "futurmaster_clean.csv", encoding='utf-8')


def clean_material(material):
    new_mat = np.where(material == "#", -1, material.str[-5:])
    new_mat = new_mat.astype(int)
    return new_mat


def clean_rebates():
    #Load data
    reb = io.read_csv(PATH_DATA_RAW, "Rebates", "Rebates1.txt", delimiter=";", skiprows=[0, 1, 3])
    files = ["Rebates/Rebates1.txt",
             "Rebates/Rebates2.txt",
             "Rebates/Rebates3.txt"]
    rebates = pd.concat(list(map(loader, files)))

    # Clean data
    rebates = clean_col_names(rebates)

    cols_to_drop = [_ for _ in rebates.columns if "unit" in _]
    rebates = rebates.drop(columns=cols_to_drop)

    rebates.material = clean_material(rebates.material)

    # Save clean data
    io.write_csv(rebates, PATH_DATA_FORMATTED, "rebates_clean.csv")


def clean_customer_data():

    df_customers = io.read_csv(PATH_DATA_RAW, 'Customers', 'CPG_SoldTo_Mapping.csv', sep=';', encoding='utf-8')
    df_customers.columns = [n.FIELD_CUSTOMER_GROUP_NAME, 'sold-to', n.FIELD_CUSTOMER]

    for col in [n.FIELD_CUSTOMER_GROUP_NAME, n.FIELD_CUSTOMER]:
        df_customers[col] = format_string_names(df_customers[col])

    df_customer_group = io.read_excel(PATH_DATA_RAW, 'Customers', 'CPG ID name mapping.xlsx', header=None)
    df_customer_group.columns = [n.FIELD_CUSTOMER_GROUP_NAME, n.FIELD_CUSTOMER_GROUP]
    df_customer_group[n.FIELD_CUSTOMER_GROUP_NAME] = format_string_names(df_customer_group[n.FIELD_CUSTOMER_GROUP_NAME])

    df_customers = (df_customers
                    .filter([n.FIELD_CUSTOMER_GROUP_NAME, n.FIELD_CUSTOMER])
                    .merge(df_customer_group, on=n.FIELD_CUSTOMER_GROUP_NAME, how='left')
                    .drop_duplicates()
                    )
    df_customers['brand'] = df_customers[n.FIELD_CUSTOMER_GROUP_NAME].str.replace('FR ', '')

    df_customers['store'] = df_customers[n.FIELD_CUSTOMER].copy()
    df_customers['store'] = df_customers['store'].str.replace(pat=r'(A[0-9]+)', repl='')

    for value in list(df_customers['brand'].unique()) + ['SARL']:
        df_customers['store'] = df_customers['store'].str.replace(value, '').str.strip()

    # df_customers['store'] = df_customers[n.FIELD_CUSTOMER].replace(regex=df_customers['brand'], value="").str.strip()

    io.write_csv(df_customers, PATH_DATA_FORMATTED, 'customer_information_cleaned.csv')


def clean_sales():
    files = glob.glob(PATH_DATA_RAW + "/SalesFR/*")
    sales = pd.concat(list(map(lambda x: io.read_csv(x, delimiter=";"), files)))

    # Clean data
    sales = clean_col_names(sales)
    sales["year"] = misc.get_year(sales.calendar_yearweek)
    sales.material = clean_lead_sku(sales.material)

    sales = sales.drop_duplicates()

    # Save clean data
    io.write_csv(sales, PATH_DATA_FORMATTED, "sales_clean.csv")


# Find geo_loc function
def get_geo_codes(address):
    key = '6j_USCas00Jqmpj2EFtKe18nLWPylj0gPTOLeBypt9U'
    apiversion = 1.0
    url = 'https://atlas.microsoft.com/search/fuzzy/json'
    parameters = {'Content-Type': 'application/json', 'subscription-key': key,
                  'api-version': apiversion, 'query': address}
    response = requests.get(url, parameters)

    return response.json()


def correct_location(j_son, country_code='FR'):
    res = j_son['results']
    found_french = False
    for i in range(len(res)):
        if res[i]['address']['countryCode'] == 'FR':
            found_french = True
            break
    if found_french:
        lat = res[i]['position']['lat']
        lon = res[i]['position']['lon']
    if not found_french:
        lat = lon = np.nan
    return lat, lon


def clean_weather():
    """
    - Reads data from different folders and create respecitve 'city' column
    - Uses the get_geo_codes() function to allocate latitude and longitude
    """
    # For now, we use the following as hardcoded. We should change that in the future.
    weather_cities = ['ajaccio', 'bordeaux', 'dijon', 'lille', 'lyon', 'marseille', 'nantes', 'orleans', 'paris',
                      'rennes', 'rouen', 'strasbourg', 'toulouse']

    weather_df = pd.DataFrame()
    for city in weather_cities:  # Easier to work with a loop here as the city is not specified in the raw tables
        path_temp = PATH_DATA_RAW + '/Weather/weather_' + city
        files = glob.glob(path_temp + "/*.csv")
        temp_df = pd.concat(list(map(lambda x: pd.read_csv(x), files)))
        temp_df['city'] = city
        weather_df = weather_df.append(temp_df)

    # Convert unix to year_week

    weather_df['time'] = weather_df['time'] + 10* 3600  # Adjusts to correct day
    weather_df['time'] = pd.to_datetime(weather_df['time'],unit='s')
    weather_df['calendar_yearweek'] = [misc.get_year_week(d) for d in weather_df['time']]

    # Get geo_loc for each cities
    w_cities_geo_codes = pd.DataFrame()
    for city in weather_df['city'].unique():
        postcode = city+", France"
        lat, lon = correct_location(get_geo_codes(postcode))
        t = {'city': [city],'lat':[lat], 'lon':[lon]}
        temp = pd.DataFrame(data=t)

        w_cities_geo_codes = w_cities_geo_codes.append(temp)
    weather_df = pd.merge(weather_df, w_cities_geo_codes, how='left', on=['city'])

    # Save clean data
    io.write_csv(weather_df, PATH_DATA_FORMATTED, "weather_clean.csv")


def clean_plants():
    plants = io.read_excel(PATH_DATA_RAW + '/FR Location addresses.xlsx', skiprows=3)[['Plant', 'Postal Code']]

    # Get geo_loc table for plants
    plants_geo_codes = pd.DataFrame()
    for index, row in plants.iterrows():
        postcode = str(row['Postal Code'])+", France"
        lat, lon = correct_location(get_geo_codes(postcode))
        t = {'plant': [row['Plant']], 'lat': [lat], 'lon': [lon]}
        temp = pd.DataFrame(data=t)
        plants_geo_codes = plants_geo_codes.append(temp)

    # Save clean data
    io.write_csv(plants_geo_codes, PATH_DATA_FORMATTED, "plants_clean.csv")


def clean_cpg_sold_to():
    cpg_sold_to = io.read_csv(PATH_DATA_RAW, "Customers/CPG_SoldTo_Mapping.csv", sep=";", encoding="utf-8")
    cpg = io.read_excel(PATH_DATA_RAW, "Customers/CPG ID name mapping.xlsx", header=None,
                        names=["customer_planning_group_name", "customer_planning_group"])

    cpg_sold_to = clean_col_names(cpg_sold_to)
    cpg_sold_to = cpg_sold_to.rename(columns={"sold-to": "sold_to_party_id",
                                              "customer_planning_group": "customer_planning_group_name"})

    cpg_sold_to = pd.merge(cpg_sold_to, cpg, on=["customer_planning_group_name"], how="left")

    io.write_csv(cpg_sold_to, PATH_DATA_FORMATTED, "cpg_sold_to_clean.csv")


def clean_consignment_stocks(country, folder_extract, years):
    results = list()
    for sheet_name in years:
        data = io.read_excel(PATH_DATA_RAW, 'ConsignmentStocks', 'ZOCI_ZOCF 2017_2019.xlsx', sheetname=sheet_name,
                             header=8)
        year_weeks = list(
            io.read_excel(PATH_DATA_RAW, 'ConsignmentStocks', 'ZOCI_ZOCF 2017_2019.xlsx', sheetname=sheet_name,
                          header=7, nrows=1).columns)[1:]
        year_weeks = [''.join(year_week.split('.')[::-1]) for year_week in year_weeks]

        data.columns = list(data.columns[: -len(year_weeks)]) + list(year_weeks)
        data.columns = ['_'.join(col.replace('.', '').lower().split()) for col in data.columns]

        # ZOCF = consignment fill up vs ZOCI = Consignment Issue (= Invoiced so customer order less usually)
        relevant_columns = [
            'sales_doc_type',
            n.FIELD_SKU_ID,
            n.FIELD_CUSTOMER_GROUP,
            n.FIELD_PLANT_ID,
        ]
        data = data[relevant_columns + year_weeks]

        data = pd.melt(data, id_vars=relevant_columns, value_vars=year_weeks,
                       var_name=n.FIELD_YEARWEEK, value_name='volume_hl_consignment')

        data = (data[data['volume_hl_consignment'].notnull()]
                .groupby(relevant_columns + [n.FIELD_YEARWEEK]).sum().reset_index())
        results.append(data)

    data = pd.concat(results, axis=0, sort=False)

    # Get customer planning group code
    customers = (io.read_csv(PATH_DATA_FORMATTED, 'cpg_sales_office.csv', sep=';')
                 .rename(columns={'Customer Planning Group': n.FIELD_CUSTOMER_GROUP,
                                  'Load Week': n.FIELD_CUSTOMER_GROUP_NAME})
                 .set_index(n.FIELD_CUSTOMER_GROUP_NAME)
                 )
    data[n.FIELD_CUSTOMER_GROUP] = data[n.FIELD_CUSTOMER_GROUP].map(customers[n.FIELD_CUSTOMER_GROUP].to_dict())
    data[n.FIELD_YEARWEEK] = data[n.FIELD_YEARWEEK].astype(int)

    io.write_csv(data, PATH_DATA_FORMATTED, country, folder_extract, 'consignment_clean.csv')


if __name__ == "__main__":

    country, folder_extract = 'france', '201918'
    clean_forecastaccuracy_bias()
    clean_forecastaccuracy_monthbias()
    clean_hist_forecast()
    clean_activities()
    clean_materials_md()
    clean_stock()
    clean_skus_uom()
    clean_rebates()
    clean_futurmaster(country, folder_extract)
    clean_weather()
    clean_plants()
    clean_consignment_stocks(country, folder_extract, ['2017', '2018', '2019'])
