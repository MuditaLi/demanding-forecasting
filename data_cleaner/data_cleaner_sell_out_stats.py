import os
import re
import pandas as pd
import itertools
import unicodedata
from dbfread import DBF
import util.input_output as io
import preprocessor.names as n
from util.paths import PATH_DATA_RAW, PATH_DATA_FORMATTED


def standardize_name(serie):
    return (serie
            .str.replace('|'.join(['-', '\'']), ' ')
            .str.upper()
            .apply(lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode())
            .str.replace('SAINT', 'ST')
            )


def quarter_as_column(tmp):
    columns_quarter = list(filter(re.compile(r'(^Q[0-9]_[0-9]{4})').search, tmp.columns))
    tmp = tmp.rename(columns=dict(zip(
        columns_quarter,
        map(lambda s: ''.join(s.split('_')[::-1]).replace('Q', '0'), columns_quarter)))
    )

    vars = list(filter(re.compile(r'(^[0-9]{6})').search, list(tmp.columns)))
    tmp = tmp.melt(id_vars=tmp.columns.difference(vars), value_vars=vars,
                   var_name='calendar_yearquarter', value_name='unemployment_rate')

    tmp['calendar_yearquarter'] = tmp['calendar_yearquarter'].astype(int)
    return tmp


def clean_population_per_city_table():
    """
    Data about population size and distribution
    Percentage of males vs females 17-39 & 40-60 %

    Source:
        https://insee.fr/fr/statistiques/3137409#consulter
    """
    file_name_input = 'base-ic-evol-struct-pop-2014-1.xls'
    file_name_output = 'stats_cities_population_clean.csv'

    relevant_columns = ['COM', 'LIBCOM']
    table = io.read_excel(PATH_DATA_RAW, 'insee', 'population', file_name_input, sheetname='IRIS', header=4)
    # names = dict(zip(table.columns, table.loc[0, :]))
    table = table.rename(columns=table.loc[0, :]).loc[1:, :]
    table = table.filter(relevant_columns +
                         [col for col in table.columns if col.startswith('P14_') or col.startswith('C14_POP15P')])
    table = table.groupby(relevant_columns).sum().reset_index()

    ages_brakets = ['0014', '1529', '3044', '4559', '6074', '75P']  # Between 0 & 14yo, ...
    columns_agg = map(''.join, itertools.product(['P14_POP'], ['', 'H']))  # Overall, Men, Women ('F')
    columns_brackets = map(''.join, itertools.product(['P14_'], ['POP', 'H'], ages_brakets))
    columns_categories = map(lambda x: '_'.join(x).strip('_'),
                             itertools.product(['C14_POP15P'], ['', 'CS1', 'CS2', 'CS3', 'CS5', 'CS6', 'CS7']))
    columns = list(columns_brackets) + list(columns_agg) + list(columns_categories)

    tmp = table[relevant_columns + columns]

    new_columns = dict(zip(columns, map(lambda x: x.replace('P14_', ''), columns)))
    tmp = tmp.rename(columns=new_columns)

    new_columns.update({'POP': 'total_population', 'POPH': 'men_population', 'POPF': 'female_population'})
    new_columns.update({'COM': n.FIELD_CITY_CODE, 'LIBCOM': n.FIELD_CITY})
    tmp = tmp.rename(columns=new_columns)

    tmp['ratio_male_population'] = tmp['men_population'] / tmp['total_population'] * 100
    for x in ages_brakets:
        tmp['_'.join(['ratio_male_population', x])] = tmp['H' + x] / tmp['POP' + x] * 100

    for x in ages_brakets:
        tmp['_'.join(['ratio_age_bracket', x])] = tmp['POP' + x] / tmp['total_population'] * 100

    tmp.drop(columns=list(map(''.join, itertools.product(['POP', 'H'], ages_brakets))), inplace=True)
    tmp[n.FIELD_CITY_CODE] = tmp[n.FIELD_CITY_CODE].astype(str).str.zfill(5)

    names = ['population_over_15yo', 'farmers', 'artisans', 'executives', 'employees', 'workers', 'pensioners']
    codes = map(lambda x: '_'.join(x).strip('_'),
                itertools.product(['C14_POP15P'], ['', 'CS1', 'CS2', 'CS3', 'CS5', 'CS6', 'CS7']))
    tmp = tmp.rename(columns=dict(zip(codes, names)))

    for col in names:
        tmp[col] = tmp[col] / tmp['total_population'] * 100

    tmp[n.FIELD_CITY] = standardize_name(tmp[n.FIELD_CITY])
    io.write_csv(tmp, PATH_DATA_FORMATTED, file_name_output)
    # TODO - LYON / PARIS / MARSEILLE DATA PER BOROUGH AVAILABLE IN TABLE BUT MATCH UNLIKELY WITH STORES


def clean_income_distribution_per_city_table():
    """
    Statistics about people's income per city (2015)
    Source:
        https://insee.fr/fr/statistiques/3560118
    """
    file_name_input = 'FILO_DEC_COM.xls'
    file_name_output = 'stats_cities_incomes_clean.csv'

    relevant_columns = {
        'CODGEO': n.FIELD_CITY_CODE,
        'LIBGEO': n.FIELD_CITY,
        'NBUC15': 'number_of_consumer_units',
        'Q115': '1st_quartile_income',
        'Q215': 'median_income',
        'Q315': '3rd_quartile_income',
        'Q3_Q1': 'gap_between_quartiles',                                  # Delta between 1 and 3 quartiles
        'PMIMP15': 'ratio_population_paying_income_taxes',                 # Percentage of families paying income taxes
        'PCHO15': 'unemployment_benefits_percentage_of_income',            # unemployment benefits
        'S80/S20': 'insee_s80s20_distribution_income',
        'GI15': 'gini_index',
    }
    table = io.read_excel(PATH_DATA_RAW, 'insee', 'unemployment', file_name_input, sheetname='ENSEMBLE', header=4)
    # names = dict(zip(table.columns, table.loc[0, :]))
    table = table.rename(columns=table.loc[0, :]).loc[1:, :]
    table = table.filter(list(relevant_columns.keys()))
    table = table.rename(columns=relevant_columns)
    table[n.FIELD_CITY] = standardize_name(table[n.FIELD_CITY])
    io.write_csv(table, PATH_DATA_FORMATTED, file_name_output)


def clean_unemployment_data_per_department():
    """
    Unemployment data per department & quarter until Q3 2018
    Source:
        https://www.insee.fr/fr/statistiques/2012804#titre-bloc-1
    """

    file_name_input = 'sl_cho_2018T3.xls'
    file_name_output = 'stats_departments_quarterly_unemployment_rates.csv'

    relevant_columns = {
        'Libellé': 'department_name',
        'Code': 'department_code',
    }
    table = (io.read_excel(PATH_DATA_RAW, 'insee', 'unemployment', file_name_input, sheetname='Département', header=2)
             .rename(columns=relevant_columns)
             .filter(regex=r'^(department_)|(_201(5|6|7|8|9)$)'))

    table = table[table['department_name'].notnull()]
    table.columns = table.columns.str.replace('T', 'Q')
    table['department_name'] = standardize_name(table['department_name'])

    table = quarter_as_column(table)
    io.write_csv(table, PATH_DATA_FORMATTED, file_name_output)


def clean_unemployment_data_per_city():
    """
    Unemployment data per employment zone & quarter until Q3 2018

    Source:
        https://www.insee.fr/fr/statistiques/2012804#titre-bloc-1

    Additional source:
        Mapping employment zone to corresponding cities
        https://www.insee.fr/fr/information/2114596
    """
    file_name_input = 'chomage-zone-t1-2003-t3-2018.xls'
    file_name_input_mapping = 'ZE2010 au 01-01-2018.xls'
    file_name_outpout = 'stats_employment_zones_quarterly_unemployment_rates.csv'

    relevant_columns = {
        'LIBZE2010': 'employment_zone_name',
        'ZE2010': 'employment_zone_code',
    }
    table = (io.read_excel(PATH_DATA_RAW, 'insee', 'unemployment', file_name_input, sheetname='txcho_ze', header=5)
             .rename(columns=relevant_columns)
             .filter(regex=r'^(employment_)|(201(5|6|7|8|9)-T[0-9]$)'))

    table = table[table['employment_zone_name'].notnull()]
    table.columns = table.columns.str.replace('T', 'Q')
    table.columns = list(map('_'.join, map(lambda x: x[::-1], table.columns.str.split('-').tolist())))
    table['employment_zone_name'] = standardize_name(table['employment_zone_name'])
    table['employment_zone_code'] = table['employment_zone_code'].str.replace('*', '')

    relevant_columns = {
        'CODGEO': n.FIELD_CITY_CODE,
        'LIBGEO': n.FIELD_CITY,
        'LIBZE2010': 'employment_zone_name',
        'ZE2010': 'employment_zone_code',
    }
    match = (io.read_excel(PATH_DATA_RAW, 'insee', 'unemployment', file_name_input_mapping,
                           sheetname='Composition_communale', header=5)
             .rename(columns=relevant_columns)
             .filter(list(relevant_columns.values()))
             )

    match[n.FIELD_CITY] = standardize_name(match[n.FIELD_CITY])
    match['employment_zone_name'] = standardize_name(match['employment_zone_name'])
    match['employment_zone_code'] = match['employment_zone_code'].astype(str).str.zfill(4)

    table = table.merge(match.drop(columns=['employment_zone_name']), on='employment_zone_code', how='left')
    table = quarter_as_column(table)
    io.write_csv(table, PATH_DATA_FORMATTED, file_name_outpout)


if __name__ == '__main__':
    clean_population_per_city_table()
    clean_income_distribution_per_city_table()
    clean_unemployment_data_per_department()
    clean_unemployment_data_per_city()
