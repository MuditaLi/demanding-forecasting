import datetime
from fbprophet import Prophet
from itertools import product
import pandas as pd
from preprocessor.data_loader import DataLoader
import preprocessor.names as n


class LongTermPredictor:

    def __init__(self, start_train, end_train, start_test, end_test, forecasting_horizons):
        """
        Object to forecast monthly volumes per SKU at long-term horizons (> 18 months) using the
        Facebook Prophet algorithm.
        start_test and end test are of no use except for backtesting
        Args:
            start_train (int): first date_to_predict to train model
            start_test (int): first date_to_predict to test model
            end_test (int): last date_to_predict to test model
            forecasting_horizons ([int): forecast horizon in months
        """

        self.start_test = start_test
        self.end_test = end_test
        self.start_train = start_train
        self.end_train = end_train
        self.forecasting_horizons = forecasting_horizons

    def build_train_set(self, data: DataLoader):
        """
        Build the training set, the labels

        Args:
            data (DataLoader): data object containing all relevant tables

        Returns:
            train (pd.DataFrame): Train set including target labels
            x_train : train set in numpy array
            y_train: labels
        """

        # building training set and labels
        fc_acc = data.df_sell_in_fc_acc

        temp = fc_acc.groupby([n.FIELD_LEAD_SKU_ID])['total_shipments_volume'].sum().reset_index()
        ciblesku = temp[temp.total_shipments_volume > 0][n.FIELD_LEAD_SKU_ID].unique()

        ciblefcacc = fc_acc[
            (fc_acc[n.FIELD_LEAD_SKU_ID].isin(ciblesku))
            &
            (fc_acc.calendar_yearmonth >= self.start_train)
            &
            (fc_acc.calendar_yearmonth <= self.end_train)
        ]

        df = ciblefcacc.groupby([n.FIELD_YEARMONTH, n.FIELD_LEAD_SKU_ID])['total_shipments_volume'].sum().reset_index()

        data = pd.DataFrame(list(product(df.calendar_yearmonth.unique(), df.lead_sku.unique())),
                            columns=[n.FIELD_YEARMONTH, n.FIELD_LEAD_SKU_ID])

        data = pd.merge(data, df, how='left', on=[n.FIELD_YEARMONTH, n.FIELD_LEAD_SKU_ID])

        data.fillna(0, inplace=True)

        return data

    def fit_and_predict(self, data: DataLoader):
        """
        Fit one Facebook Prophet model per Lead SKU, and predict all sales volumes for the next months up to
        the specified forecasting_horizon.

        Args:
            data (DataLoader): data object containing all relevant tables

        Returns:
            res (pd.DataFrame): Table containing all predictions (one per lead SKU) of monthly sales for the specified
            time horizon.
        """

        x_train = self.build_train_set(data)
        res = pd.DataFrame()
        for lead_sku_id in x_train.lead_sku.unique():
            train_sku = x_train[x_train[n.FIELD_LEAD_SKU_ID] == lead_sku_id]
            df = pd.DataFrame(columns=['ds', 'y'])
            df['ds'] = train_sku[n.FIELD_YEARMONTH].values
            df['y'] = train_sku['total_shipments_volume'].values
            df.ds = df.ds.apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m"))

            # Instantiate & Fit Prophet model
            model = Prophet(daily_seasonality=False, weekly_seasonality=False)

            # hyper parameter of prophet model can be more refined
            model.add_seasonality(name='monthly', period=4, fourier_order=5)
            model.fit(df)

            # Predict future sales
            future_data = model.make_future_dataframe(periods=self.forecasting_horizons[0], freq='m')
            forecast_data = model.predict(future_data)
            predictions = forecast_data.yhat[-self.forecasting_horizons[0]:]
            datetopred = (forecast_data.ds + pd.DateOffset(1)).apply(
                lambda x: x.strftime('%Y%m')).values[-self.forecasting_horizons[0]:]

            temp = pd.DataFrame()
            temp['predictions'] = predictions
            temp['date_to_predict'] = datetopred
            temp['date_when_predicting'] = self.start_test
            temp[n.FIELD_LEAD_SKU_ID] = lead_sku_id
            res = pd.concat([res, temp])

        return res
