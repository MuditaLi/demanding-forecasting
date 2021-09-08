import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from features.pipeline_helpers import TypeSelector, ColumnSelector, clean_ohe_cols


def features_materials(materials, is_training, materials_pipeline=None):
    """
    Creates a DataFrame containing material-related features. It will impute missing values, one-hot encode categorical
    variables.
    :param materials: pd.DF containing material-related raw data
    :param is_training: bool, are we using for training or test data
    :param materials_pipeline: either None if is_training or an already fit pipeline if is_training is False
    :return:
        - features: pd.DF containing material-related features, indexed on lead_sku
        - materials_pipeline: sklearn pipeline that has been used to transform `materials` into `features`
    """
    if is_training:
        materials_pipeline = make_materials_pipeline()
        materials_pipeline.fit(materials)
    else:
        assert isinstance(materials_pipeline, Pipeline), \
            "When is_training is False, materials_pipeline needs to be input."

    features = pd.DataFrame(data=materials_pipeline.transform(materials),
                            columns=get_materials_columns(materials_pipeline))

    features = features.rename(columns={'material': 'lead_sku'})

    return features, materials_pipeline


def make_materials_pipeline():
    preprocess_pipeline = make_pipeline(
        FeatureUnion(transformer_list=[
            ("numeric_features", make_pipeline(
                TypeSelector(np.number),
                SimpleImputer(strategy="median"),
                # StandardScaler()
                # if we want to use it, we need to first remove material using
                # column_selector and add it back with a feature union
            )),
            ("categorical_features", make_pipeline(
                TypeSelector("category"),
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(sparse=False, )
            ))
        ])
    )
    return preprocess_pipeline


def get_materials_columns(materials_pipeline):
    num_cols = materials_pipeline.named_steps["featureunion"] \
        .transformer_list[0][1] \
        .named_steps["typeselector"] \
        .get_feature_names()
    num_cols = list(num_cols)

    cat_cols = materials_pipeline.named_steps["featureunion"] \
        .transformer_list[1][1] \
        .named_steps["typeselector"] \
        .get_feature_names()

    ohe_cols = materials_pipeline.named_steps["featureunion"] \
        .transformer_list[1][1] \
        .named_steps["onehotencoder"] \
        .get_feature_names()
    ohe_cols = clean_ohe_cols(ohe_cols, cat_cols)

    all_cols = num_cols + ohe_cols
    return all_cols

