import pandas as pd
import numpy as np
import preprocessor.names as n
import itertools


def create_dummy_var(data, list_groups, use_nbr_activities=True):
    """
    Args:
        use_nbr_activities (bool): Whenever True, we use the true number of activities given in the
        'no_of_activities' column """
    if not use_nbr_activities:
        for var in list_groups:
            data[var.replace(" ", "_").lower()] = np.where(data['category_name'] == var, 1, 0)
        data['other_visit'] = np.where(~data['category_name'].isin(list_groups), 1, 0)
    else:
        for var in list_groups:
            data[var.replace(" ", "_").lower()] = np.where(data['category_name'] == var, data['no_of_activities'], 0)
        data['other_visit'] = np.where(~data['category_name'].isin(list_groups), data['no_of_activities'], 0)

    return data.drop(columns=['category_name', 'no_of_activities'])


# Create windows range of lookup around dates_to_predict:
def generate_windows(data, month_ranges, key=n.FIELD_CUSTOMER_GROUP):
    columns_to_ignore = [n.FIELD_CUSTOMER_GROUP, n.FIELD_YEARMONTH]
    original_columns = [c for c in list(data.columns) if c not in columns_to_ignore]  # list columns to create
    data = data.sort_values(by=columns_to_ignore)  # Convenient to use shift function
    for i in month_ranges:
        for col in original_columns:
            data['_month'.join([col, str(i)])] = data.groupby([key])[col].shift(periods=(-1*i))
    data = data.drop(original_columns, axis=1)
    return data


# Get keys
def build_salesforce_activity_features(activities, fc_acc, dates_when_predicting):
    """
    Creates a DataFrame containing all relevant activities for the last three months on a cpg level.
    :param activities: curated - 'activities_clean.csv'
    :param fc_acc: curated - 'fc_acc_week.csv' (any table with a full yearweek to yearmonth would work
    :param dates_when_predicting: list of dates for which we want prediction
    """
    # Create dummy variables & group by month
    df = create_dummy_var(activities, ['Business Review', 'Sales visit'], use_nbr_activities=True)
    df = df.groupby([n.FIELD_YEARMONTH, n.FIELD_CUSTOMER_GROUP]).sum().reset_index()

    # Add rows with zeros for missing time steps
    df_full = pd.DataFrame(list(itertools.product(df[n.FIELD_CUSTOMER_GROUP].unique(), df[n.FIELD_YEARMONTH].unique())),
                           columns=[n.FIELD_CUSTOMER_GROUP, n.FIELD_YEARMONTH])
    df = pd.merge(df_full, df, how='left', on=[n.FIELD_CUSTOMER_GROUP, n.FIELD_YEARMONTH]).fillna(0)

    # Create columns for three previous months
    df = generate_windows(df, [-3, -2, -1])

    # Fills missing values for the first 3 months
    # Here we use most_frequent obs, which turns out to be 0 for each of the 3 columns considered
    df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

    # Filter for appropriate weeks
    month_week_mapping = (fc_acc
                          .filter([n.FIELD_YEARMONTH, n.FIELD_YEARWEEK])
                          .drop_duplicates(subset=[n.FIELD_YEARWEEK]))

    weeks_to_use = (pd.DataFrame(data={n.FIELD_YEARWEEK: dates_when_predicting})
                    .merge(month_week_mapping, on=[n.FIELD_YEARWEEK], how='left'))

    df_salesforce_features = (weeks_to_use
                              .merge(df, on=[n.FIELD_YEARMONTH], how='left')
                              .drop(columns=[n.FIELD_YEARMONTH]))

    return df_salesforce_features.rename(columns={n.FIELD_YEARWEEK: 'date_when_predicting'})
