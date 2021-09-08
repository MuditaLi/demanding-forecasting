from itertools import product
import pandas as pd
import preprocessor.names as n
from functools import partial
import util.misc as misc


# Compute distances to closest rebates in past & future
def compute_time_delta(df, start, end):
    return df[[end, start]].apply(
        lambda x: partial(misc.delta_between_periods, is_weekly=True)(*x), axis=1)


def find_closest_inactive_rebate(df_rebates_list, granularity_cols, max_horizon: int=32):

    # Weeks since last rebate
    columns_selection = [n.FIELD_YEARWEEK + '_end', n.FIELD_REBATE_AMOUNT, n.FIELD_REBATE_RATE]
    last_rbt = (df_rebates_list[(df_rebates_list[n.FIELD_YEARWEEK + '_end'] < df_rebates_list['date_to_predict'])]
                .sort_values(columns_selection, ascending=[False, True, True])
                .drop_duplicates(granularity_cols)
                )
    if not last_rbt.empty:
        last_rbt['weeks_to_rebate'] = compute_time_delta(last_rbt, 'date_to_predict', n.FIELD_YEARWEEK + '_end')

    # Weeks to next_rebate
    columns_selection = [n.FIELD_YEARWEEK + '_start', n.FIELD_REBATE_AMOUNT, n.FIELD_REBATE_RATE]
    next_rbt = (df_rebates_list[(df_rebates_list[n.FIELD_YEARWEEK + '_start'] > df_rebates_list['date_to_predict'])]
                .sort_values(columns_selection, ascending=[True, True, True])
                .drop_duplicates(granularity_cols)
                )
    if not next_rbt.empty:
        next_rbt['weeks_to_rebate'] = compute_time_delta(next_rbt, 'date_to_predict', n.FIELD_YEARWEEK + '_start')

    inactive_rebates = pd.concat([last_rbt, next_rbt], axis=0, sort=False)
    inactive_rebates['filter'] = inactive_rebates['weeks_to_rebate'].abs()
    inactive_rebates = (inactive_rebates.sort_values('filter', ascending=False)
                        .drop_duplicates(granularity_cols))

    # Create binary variable to indicate whether there is an active rebate contract for the week to predict
    inactive_rebates['is_on_rebate'] = 0

    return inactive_rebates[inactive_rebates['filter'] <= max_horizon].drop(columns=['filter'])


def find_most_relevant_active_rebate(df_rebates_list, granularity_cols):
    # Keep rows with valid rebate
    active_rebates = df_rebates_list[
        (df_rebates_list['date_to_predict'] >= df_rebates_list[n.FIELD_YEARWEEK + '_start'])
        &
        (df_rebates_list['date_to_predict'] <= df_rebates_list[n.FIELD_YEARWEEK + '_end'])
    ]

    # Aggregation strategy: Keep rebate with highest reward whenever there is several entries for same SKU / CPG
    active_rebates.sort_values([n.FIELD_REBATE_AMOUNT, n.FIELD_REBATE_RATE], ascending=True, inplace=True)
    active_rebates = active_rebates.drop_duplicates(granularity_cols)

    # Create binary variable to indicate whether there is an active rebate contract for the week to predict
    active_rebates['is_on_rebate'] = 1

    delta_weeks = partial(compute_time_delta, df=active_rebates, end=n.FIELD_YEARWEEK + '_end')

    # Number of weeks left with active rebates contract
    active_rebates['weeks_to_rebate'] = delta_weeks(start='date_to_predict')

    return active_rebates


def build_rebates_features(dates_when_predicting: list, dates_to_predict: list,
                           df_combinations_to_predict: pd.DataFrame, rebates: pd.DataFrame) -> pd.DataFrame:

    # Solve issue with weeks 53 (wrong time difference computations otherwise)
    for col in [n.FIELD_YEARWEEK + '_start', n.FIELD_YEARWEEK + '_end']:
        rebates[col] = rebates[col].where(rebates[col] % 100 <= 52, rebates[col] - 1)

    # Rebate contract length
    rebates['rebate_length_weeks'] = compute_time_delta(rebates, n.FIELD_YEARWEEK + '_start', n.FIELD_YEARWEEK + '_end')

    # Exclude long-term contracts (no promotion / variable cost effect)
    rebates = rebates[~rebates[n.FIELD_CUSTOMER_GROUP].isin(['F30', 'F31'])]
    rebates = rebates[rebates['rebate_length_weeks'] <= 30]

    # TODO - CLARIFY WHEN REBATES DATA IS AVAILABLE (SO THAT WE CAN USE IT TO PREDICT)
    rebates[n.FIELD_YEARWEEK + '_first_seen'] = rebates[n.FIELD_YEARWEEK + '_start'].apply(
        partial(misc.substract_period, add=6, highest_period=52))

    # Binary variable for type of rebate (mechanism is either volume or value related)
    rebates['is_volume_rebate'] = (rebates[n.FIELD_REBATE_MECHANISM] == 'volume').astype(int)

    granularity_cols = [
        n.FIELD_CUSTOMER_GROUP,
        n.FIELD_LEAD_SKU_ID,
        'date_when_predicting',
        'date_to_predict'
    ]

    # Table containing all combinations to predict
    df_features = df_combinations_to_predict.filter([n.FIELD_CUSTOMER_GROUP, n.FIELD_LEAD_SKU_ID]).drop_duplicates()
    df_features = product(df_features.to_records(index=False), zip(dates_when_predicting, dates_to_predict))
    df_features = map(lambda x: tuple(list(x[0]) + list(x[1])), df_features)
    df_features = pd.DataFrame(list(df_features), columns=granularity_cols)

    # Merge with rebates table and filter out all rebates info that are not known at time of prediction
    df_features_full = df_features.merge(rebates, on=[n.FIELD_LEAD_SKU_ID, n.FIELD_CUSTOMER_GROUP], how='inner')
    df_features_full = df_features_full[
        (df_features_full['date_when_predicting'] >= df_features_full[n.FIELD_YEARWEEK + '_first_seen'])
    ]

    inactive_rebates = find_closest_inactive_rebate(df_features_full, granularity_cols)
    active_rebates = find_most_relevant_active_rebate(df_features_full, granularity_cols)

    df_features = (pd.concat([inactive_rebates, active_rebates], axis=0, sort=False)
                   .sort_values('is_on_rebate', ascending=False)
                   .drop_duplicates(granularity_cols)
                   )

    feature_columns = [
        'rebate_length_weeks',
        'is_on_rebate',
        'is_volume_rebate',
        'weeks_to_rebate',
        n.FIELD_REBATE_RATE,
        # n.FIELD_REBATE_TARGET,
    ]

    print(df_features.shape)

    return df_features[granularity_cols + feature_columns]
