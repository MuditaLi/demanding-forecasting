"""
This module contains functions to save and load data.
"""

import logging
import os.path
import json
import pickle
import yaml

import pandas as pd

from util.misc import optimize_memory

logging.basicConfig(
    # filename=os.path.join(PATH_OUTPUT, 'import.log'),
    format='%(levelname)s %(asctime)s %(message)s',
    level=logging.DEBUG
)

DTYPES = {"week": int,
          "Actual Sales Volume": float,
          "Alcohol percentage": float,
          "Brand": str,
          "Calendar Year/Month": int,
          "Calendar year/week": int,
          "Component scrap in percent": float,
          "Company (as-is)": str,
          "Customer": str,
          "Customer Group (As-Is)": str,
          "Customer Planning Group": str,
          "Customer Status": str,
          "Emp Resp (As-was)": str,
          "Forecast Accuracy % (w-1)": float,
          "Forecast Accuracy % (w-4)": float,
          "Forecast Bias (w-1)": float,
          "Forecast Bias (w-4)": float,
          "Forecast volume M-3": float,
          "Global Beverage Category": str,
          "Global Beverage Cat": str,
          "global_beverage_cat": str,
          "Global Chain(as-is)": str,
          "InhseProdTime": str,
          "KAM (As-was)": str,
          "Lead SKU": str,
          "Material": str,
          "Material ID": str,
          "Material group 1": str,
          "Material Local Status": str,
          "Material Local Status (as-is)": str,
          "Material Local Status (as-was)": str,
          "Material Pack Size": str,
          "Material Pack Type": str,
          "Own/Foreign Brand": str,
          "Order Item Creation Date": str,
          "Payer": str,
          "Produit Code EAN PAL": str,
          "Produit Code EAN UC": str,
          "Produit Code SKU": int,
          "Plant": str,
          "Product Category": str,
          "Reason for rejection": str,
          "Sales Dist. (As-was)": str,
          "Sales Document No": str,
          "Sales Forecast volume (w-1)": float,
          "Sales Forecast volume (w-4)": float,
          "Sales Group": str,
          "Sales Office": str,
          "Sales group (As-was)": str,
          "Ship-To": str,
          "Shipping point": str,
          "Sold-to party": str,
          "Statist. Curr.": str,
          "Storage location": str,
          "Sub Brand": str,
          "Trade Channel Level 1": str,
          "Total Shipments Volume": float,
          "Unit": str,
          "Unit.1": str,
          "Unit.2": str,
          "Unit.3": str,
          "Unit.4": str,
          "Unit.5": str,
          }


def construct_path(*path):
    path = os.path.join(*path)
    path = os.path.join(os.path.dirname(__file__), path) if not os.path.isabs(path) else path
    return path


def append_csv(dataframe, *path):
    path = construct_path(*path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    dataframe.to_csv(path, index=False, mode='a', header=False)


def read_csv(*path, encoding='latin1', **kwargs):
    """
    Loads a dataframe from csv file, correctly attributing dtypes based on column names
    and setting a consistent hierarchy of indices.

    Args:
        path: location of the csv file

    Returns:
        pandas.DataFrame
    """
    path = construct_path(*path)
    try:
        df = pd.read_csv(path, dtype=DTYPES, encoding=encoding, na_values=["#"], **kwargs)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
        # log(f'Empty dataframe found in {path}', os.path.basename(__file__))
        logging.info(f'Empty dataframe found in {path}', os.path.basename(__file__))

    return df


def write_csv(dataframe: pd.DataFrame, *path, **kwargs):
    """
    Writes a DataFrame to disk as csv file, maintaining project standards.

    Args:
        DataFrame (pandas.DataFrame): the DataFrame to save
        path: file location
    """
    path = construct_path(*path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataframe.to_csv(path, index=False, **kwargs)


def read_yaml(*path):
    """
    Creates a dictionary from a YALM file
    """
    # Read YAML file
    path = construct_path(*path)
    with open(path, 'r') as stream:
        data_loaded = yaml.load(stream)
    return data_loaded


def write_yaml(dictionary, *path):
    """
    Write dictionary as a YAML file
    """
    path = construct_path(*path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w', encoding='utf8') as outfile:
        yaml.dump(dictionary, outfile, default_flow_style=False, allow_unicode=True)
    return path


def write_json(dict, *path):
    """
    Writes a dictionary to disk as json file, maintaining project standards.

    Args:
        dict (dictionary): the dictionary to save
        path: file location
    """
    path = construct_path(*path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    file_pointer = open(path, 'w')
    file_pointer.write(json.dumps(dict, indent=4))
    file_pointer.close()


def read_hdf(*path, **kwargs):
    """
    Loads a DataFrame from hdf5 file, correctly attributing dtypes based on column names
    and setting a consistent hierarchy of indices.

    Args:
        path: location of the hdf5 file

    Returns:
        pandas.DataFrame
    """
    path = construct_path(*path)
    df = pd.read_hdf(path, **kwargs)

    df = optimize_memory(df)
    return df


def get_hdf_len(*path):
    """
    Returns the number of rows in an hdf file as an int.
    """
    path = construct_path(*path)
    with pd.HDFStore(path) as store:
        numrows = store.get_storer('data').nrows
    return numrows


def write_hdf(dataframe, *path):
    """
    Writes a dataframe to disk as hdf file, maintaining project standards.

    Args:
        dataframe (pandas.DataFrame): the dataframe to save
        path: file location
    """
    path = construct_path(*path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    dataframe.to_hdf(path, key='data', mode='w', complevel=9, complib='blosc', format='table')


def append_hdf(dataframe, *path):
    """
    Append a dataframe to an existing hdf file.

    Args:
        dataframe (pandas.DataFrame): the dataframe to save
        path: file location
    """
    path = construct_path(*path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    dataframe.to_hdf(path, key='data', mode='a', complevel=9, complib='blosc', format='table', append=True)


def write_string(string, *path):
    """
    Writes a string to a text file.

    Args:
        string (str)
    """
    path = construct_path(*path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(string)


def read_string(*path, clear_EOL=False):
    """
    Read a text file

    Args:
        string (str)
    Returns:
        list: each line of the string as one element in the list
    """
    path = construct_path(*path)
    with open(path, 'r') as f:
        output = f.readlines()
    if clear_EOL:
        output = [el.replace("\n", "") for el in output]
    return output


def read_excel(*path, sheetname='Sheet1', **kwargs):
    """
    Read an excel file

    Args:
        path (str): location of the file
        sheetname (str): name of sheet to be loaded
    Returns:
        pandas.DataFrame
    """

    path = construct_path(*path)
    df = pd.read_excel(path, sheetname=sheetname, dtype=DTYPES, **kwargs)

    return df


def write_pickle(obj, *path):
    """
    Filepath in .pkl
    """
    path = construct_path(*path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_pickle(*path):
    path = construct_path(*path)
    with open(path, 'rb') as f:
        return pickle.load(f)
