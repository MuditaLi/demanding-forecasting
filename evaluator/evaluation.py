import numpy as np
import pandas as pd
from util.scores import carlsberg_score, smape, bias_score, root_mean_squared_error
import preprocessor.names as n
from preprocessor.data_loader import DataLoader
from util.logger import Logger


class BaseEvaluator:

    FIELD_CARLSBERG_ACCURACY = 'Carlsberg_accuracy'
    FIELD_CARLSBERG_BIAS = 'Bias'
    FIELD_WEIGHTED_SMAPE = 'Weighted_SMAPE'
    FIELD_RMSE = 'RMSE'

    @staticmethod
    def format_metrics(carlsberg_score_value, carlsberg_bias_value, weighted_smape_value, rmse_value):
        text = [f"{BaseEvaluator.FIELD_CARLSBERG_ACCURACY} = {carlsberg_score_value*100:.0f}%",
                f"{BaseEvaluator.FIELD_CARLSBERG_BIAS} = {carlsberg_bias_value*100:.1f}%",
                f"{BaseEvaluator.FIELD_WEIGHTED_SMAPE} = {100-weighted_smape_value * 100:.1f}%",
                f"{BaseEvaluator.FIELD_RMSE} = {rmse_value:.2f}"]

        return ' | '.join(text)

    @staticmethod
    def compute_metrics(data_granularity, column_predictions, column_ground_truth):
        actuals = np.array(data_granularity[column_ground_truth])
        predictions = np.array(data_granularity[column_predictions])
        carls_score = carlsberg_score(actuals, predictions)
        carls_bias = bias_score(actuals, predictions)
        smape_ = smape(predictions, actuals)
        weight_smape = smape(predictions, actuals, actuals)
        rmse_ = root_mean_squared_error(predictions, actuals)

        return carls_score, carls_bias, smape_, weight_smape, rmse_


class Evaluator(BaseEvaluator):

    GROUND_TRUTH_COLUMN_APO_FINAL_FORECAST_W1_W4 = 'total_shipments_volume'
    GROUND_TRUTH_COLUMN_APO_FINAL_FORECAST_M3 = 'actual_sales_volume'
    GROUND_TRUTH_SELL_IN_SALES = 'total_shipments_volume'

    GROUND_TRUTH_APO_FORECAST_ACCURACY_REPORT_M3 = 'fc_acc_month.csv'
    GROUND_TRUTH_APO_FORECAST_ACCURACY_REPORT_W1_w4 = 'fc_acc_week.csv'

    GROUND_TRUTH_SELL_OUT_QTY = 'label'  # n.FIELD_QTY_SOLD_SU

    ground_truth_column = None

    def __init__(self, data: DataLoader, granularity: list, start_periods_eval: int, end_periods_eval: int,
                 forecasting_horizon: int, granularity_eval: str,
                 is_sell_in_model: bool=True, is_weekly_forecast: bool=True):

        self.data = data
        self.granularity = granularity
        self.is_sell_in_model = is_sell_in_model
        self.is_weekly_forecast = is_weekly_forecast

        self.start_eval = start_periods_eval
        self.end_eval = end_periods_eval
        self.granularity_eval = granularity_eval

        if is_sell_in_model:
            if self.granularity_eval is None:
                self.ground_truth_column = Evaluator.GROUND_TRUTH_SELL_IN_SALES
                self.data.df_fc_acc = data.df_sell_in_fc_acc
                pass
            elif self.granularity_eval == "forecast_volume_m-3":
                # Load data and column name for accuracy comparison with historical forecasts
                self.ground_truth_column = Evaluator.GROUND_TRUTH_COLUMN_APO_FINAL_FORECAST_M3
                self.data.load_sell_in_forecast_accuracy_comparison_data(
                    Evaluator.GROUND_TRUTH_APO_FORECAST_ACCURACY_REPORT_M3
                )
            else:
                self.ground_truth_column = Evaluator.GROUND_TRUTH_COLUMN_APO_FINAL_FORECAST_W1_W4
                self.data.load_sell_in_forecast_accuracy_comparison_data(
                    Evaluator.GROUND_TRUTH_APO_FORECAST_ACCURACY_REPORT_W1_w4
                )

            self.aggregations = {
                self.ground_truth_column: 'sum',
                n.FIELD_PREDICTION: 'sum'
            }

            if self.granularity_eval is not None:
                self.aggregations.update({self.granularity_eval: 'sum'})
        else:
            self.ground_truth_column = Evaluator.GROUND_TRUTH_SELL_OUT_QTY
            self.aggregations = {
                self.ground_truth_column: 'sum',
                n.FIELD_PREDICTION: 'sum'
            }

        model_type = 'sell_in' if self.is_sell_in_model else 'sell-out'
        horizon = 'w' if is_weekly_forecast else 'm'
        prints = (horizon, forecasting_horizon, model_type, 'weeks' if is_weekly_forecast else 'months')
        Logger.info('Prediction at %s-%s of %s volumes for %s' % prints,
                    '[%d, %d]' % (self.start_eval, self.end_eval), self.__class__.__name__)

    def merge_sell_in_predictions_with_historical_forecasts(
            self, model_predictions: pd.DataFrame, cpg_trade_channels: pd.DataFrame=None,
            skus_master_data: pd.DataFrame=None):
        """
        Args:
            model_predictions (pd.DataFrame): output of predict method of predictor object (forecasted volumes)
            cpg_trade_channels (pd.DataFrame): mapping between cpg code and trade channel type (ON-TRADE vs OFF-TRADE)
            skus_master_data (pd.DataFrame): brand & sub-brand allocation of a lead SKU

        Returns:
            df_combined_historical_fc_and_predictions (pd.DataFrame): predictions combined with actual historical
            forecasts made by Demand Planner (from SAP APO report)
        """
        in_period = (self.data.df_fc_acc[n.FIELD_YEARWEEK] >= self.start_eval) & \
                    (self.data.df_fc_acc[n.FIELD_YEARWEEK] <= self.end_eval)

        agg_fcts = {self.ground_truth_column: 'sum'}
        if self.granularity_eval is not None:
            agg_fcts.update({self.granularity_eval: 'sum'})

        df_hist = (self.data.df_fc_acc[in_period]
                   .groupby(self.granularity, as_index=False)
                   .agg(agg_fcts)
                   )

        mod_output = (model_predictions.copy()
                      .rename(columns={'date_to_predict': n.FIELD_YEARWEEK})
                      .filter(self.granularity + [n.FIELD_PREDICTION])
                      )
        mod_output.loc[mod_output[n.FIELD_PREDICTION] < 1, n.FIELD_PREDICTION] = 0

        df_combined_historical_fc_and_predictions = pd.merge(df_hist, mod_output, on=self.granularity, how='left')

        # TODO - Check why some values are null
        df_combined_historical_fc_and_predictions = df_combined_historical_fc_and_predictions[
            ~df_combined_historical_fc_and_predictions.isnull().any(axis=1)
        ]

        if cpg_trade_channels is not None:
            cpg_trade_channels[n.FIELD_IS_ON_TRADE] = cpg_trade_channels[n.FIELD_CUSTOMER_TRADE_CHANNEL] == 'On Trade'
            cpg_trade_channels.rename(columns={'Sales Office': 'sales_office'}, inplace=True)
            df_combined_historical_fc_and_predictions = \
                (df_combined_historical_fc_and_predictions
                 .merge((cpg_trade_channels[[n.FIELD_CUSTOMER_GROUP, n.FIELD_IS_ON_TRADE, 'sales_office']]
                         .drop_duplicates(n.FIELD_CUSTOMER_GROUP)),
                        on=n.FIELD_CUSTOMER_GROUP, how='left')
                 )

        if skus_master_data is not None:
            skus_master_data = skus_master_data[[n.FIELD_LEAD_SKU_ID, 'brand_name', 'sub_brand_name']]
            df_combined_historical_fc_and_predictions = \
                (df_combined_historical_fc_and_predictions
                 .merge(skus_master_data.drop_duplicates(n.FIELD_LEAD_SKU_ID), on=n.FIELD_LEAD_SKU_ID, how='left')
                 )

        return df_combined_historical_fc_and_predictions

    @staticmethod
    def correct_outputs(model_predictions):

        # TODO - Add classifier to find all zero values automatically
        return model_predictions.rename(columns={'date_to_predict': n.FIELD_YEARMONTH})

    def eval(self, model_predictions: pd.DataFrame,
             cpg_trade_channels: pd.DataFrame=None, skus_master_data: pd.DataFrame=None):
        """
        Evaluate the output accordingly
        """
        if self.is_sell_in_model and self.is_weekly_forecast:
            return self.merge_sell_in_predictions_with_historical_forecasts(
                model_predictions, cpg_trade_channels, skus_master_data)
        else:
            return self.correct_outputs(model_predictions)

    @staticmethod
    def _remove_main_features(granularity):
        if len(granularity) != 3:
            return

        def helper(x):
            return (x != n.FIELD_LEAD_SKU_ID and
                    x != n.FIELD_SKU_EAN and
                    x != n.FIELD_YEARWEEK and
                    x != n.FIELD_YEARMONTH)

        return list(filter(helper, granularity))[0]

    def report(self, results, granularities):
        """
        Function that will print results for different granularities
        :param results: dataframe containing the predictions
        :param granularities: the list of granularities for which we want an output
        :return:
        """
        text = ''
        for agg in granularities:
            text += '\n\n' + ' - '.join(agg)
            temp = results.groupby(agg, as_index=False).agg(self.aggregations)

            if self.granularity_eval in self.aggregations:
                hist_carls_score, hist_carls_bias, hist_smape, hist_weight_smape, hist_rmse = \
                    self.compute_metrics(temp, self.granularity_eval, self.ground_truth_column)
                text += '\n{:17s}: '.format('Historical model')
                text += self.format_metrics(hist_carls_score, hist_carls_bias, hist_weight_smape, hist_rmse)

            model_carls_score, model_carls_bias, model_smape, model_weight_smape, model_rmse = \
                self.compute_metrics(temp, n.FIELD_PREDICTION, self.ground_truth_column)

            text += '\n{:17s}: '.format('M.L. model')
            text += self.format_metrics(model_carls_score, model_carls_bias, model_weight_smape, model_rmse)

        Logger.info(text, class_name=self.__class__.__name__)

    def _individual_detailed_report(self, temp: pd.DataFrame, feature: str):
        names = [
            'value',
            'model',
            BaseEvaluator.FIELD_CARLSBERG_ACCURACY,
            BaseEvaluator.FIELD_CARLSBERG_BIAS,
            BaseEvaluator.FIELD_WEIGHTED_SMAPE,
            BaseEvaluator.FIELD_RMSE,
        ]
        frame = list()

        for group, values in temp.groupby(feature).aggregate(list).iterrows():
            metrics = list(self.compute_metrics(values, n.FIELD_PREDICTION, self.ground_truth_column))
            metrics = metrics[:2] + metrics[3:]
            frame.append(pd.DataFrame(dict(zip(names, [group, 'ML'] + metrics)), index=[0]))

            if self.granularity_eval in self.aggregations:
                metrics = list(self.compute_metrics(values, self.granularity_eval, self.ground_truth_column))
                metrics = metrics[:2] + metrics[3:]
                frame.append(pd.DataFrame(dict(zip(names, [group, 'APO'] + metrics)), index=[0]))

        tmp = pd.concat(frame, ignore_index=True)
        tmp['feature'] = feature

        # Quantity of interest is 1 - SMAPE (accuracy point of view instead of error)
        tmp[BaseEvaluator.FIELD_WEIGHTED_SMAPE] *= -1
        tmp[BaseEvaluator.FIELD_WEIGHTED_SMAPE] += 1

        if feature == n.FIELD_IS_ON_TRADE:
            tmp['value'] = np.where(tmp['value'], 'ON-TRADE', 'OFF-TRADE')

        return tmp.filter(['feature'] + names)

    def detailed_report(self, results, granularities):

        detailed_report = list()
        for agg in granularities:
            feature = self._remove_main_features(agg)
            if feature is not None:
                temp = results.groupby(agg, as_index=False).agg(self.aggregations)
                detailed_report.append(self._individual_detailed_report(temp, feature))

        detailed_report = (pd.concat(detailed_report, ignore_index=True)
                           .groupby(['feature', 'value', 'model'], sort=True)
                           .sum(min_count=1))

        relevant_columns = [
            BaseEvaluator.FIELD_CARLSBERG_ACCURACY,
            BaseEvaluator.FIELD_CARLSBERG_BIAS,
            BaseEvaluator.FIELD_WEIGHTED_SMAPE
        ]
        detailed_report[relevant_columns] = (100. * detailed_report[relevant_columns]).round(1).astype(str) + '%'

        detailed_report = '\n' + detailed_report.round(1).to_string()

        Logger.info(detailed_report, class_name=self.__class__.__name__)
