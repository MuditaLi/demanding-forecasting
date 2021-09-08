import numpy as np
import pandas as pd
import util.misc as misc
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from features.pipeline_helpers import TypeSelector, ColumnSelector, clean_ohe_cols


def features_plant(plant, is_training, plant_cpg_pipeline=None):
    """
    Creates a DataFrame containing plant and cpg-related features. It will one-hot encode categorical
    variables.
    :param plant: pd.DF containing material-related raw data
    :param is_training: bool, are we using for training or test data
    :param plant_cpg_pipeline: either None if is_training or an already fit pipeline if is_training is False
    :return:
        - features: pd.DF containing plant-related features, indexed on plants
        - plant_cpg_pipeline: sklearn pipeline that has been used to transform `materials` into `features`
    """
    plant = plant[['plant', 'customer_planning_group']].drop_duplicates()
    plant = misc.convert_categorical(plant, is_training)

    if is_training:
        plant_cpg_pipeline = make_plants_pipeline()
        plant_cpg_pipeline.fit(plant)
    else:
        assert isinstance(plant_cpg_pipeline, Pipeline), \
            "When is_training is False, plant_cpg_pipeline needs to be input."

    features = pd.DataFrame(data=plant_cpg_pipeline.transform(plant),
                            columns=get_plants_columns(plant_cpg_pipeline))
    a = pd.concat([plant.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    return a, plant_cpg_pipeline


def features_cpg(cpg, is_training, plant_cpg_pipeline=None):
    """
    Creates a DataFrame containing plant and cpg-related features. It will one-hot encode categorical
    variables.
    :param cpg: pd.DF containing material-related raw data
    :param is_training: bool, are we using for training or test data
    :param plant_cpg_pipeline: either None if is_training or an already fit pipeline if is_training is False
    :return:
        - features: pd.DF containing plant-related features, indexed on plants
        - plant_cpg_pipeline: sklearn pipeline that has been used to transform `materials` into `features`
    """
    if is_training:
        plant_cpg_pipeline = make_cpg_pipeline()
        plant_cpg_pipeline.fit(cpg)
    else:
        assert isinstance(plant_cpg_pipeline, Pipeline), \
            "When is_training is False, plant_cpg_pipeline needs to be input."

    features = pd.DataFrame(data=plant_cpg_pipeline.transform(cpg),
                            columns=get_cpg_columns(plant_cpg_pipeline))

    return features, plant_cpg_pipeline


def make_plants_pipeline():
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


def get_plants_columns(plant_cpg_pipeline):

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


def make_cpg_pipeline():
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


def get_cpg_columns(plant_cpg_pipeline):

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

