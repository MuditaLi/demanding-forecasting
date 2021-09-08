import pandas as pd
import datetime
from functools import reduce
import util.misc as misc
import numpy as np


def incorporate_sell_out(sell_out, dates_to_predict, rangemonth=2):
    """

    :param sell_out: data frame with volume sold in sell out per sku
    :param dates_to_predict:
    :param rangemonth: number of month before and after the date to predict
    :return:
    """
    res = pd.DataFrame()
    for w in dates_to_predict:
        currmonth = datetime.datetime.strptime(str(w) + '-1', "%Y%W-%w").month
        curryear = datetime.datetime.strptime(str(w) + '-1', "%Y%W-%w").year
        monthcurrent = 100*curryear + currmonth
        monthbefore = misc.substract_period(monthcurrent, rangemonth, 12)
        monthafter = misc.add_period(monthcurrent, rangemonth, 12)
        list_period = misc.create_list_period(monthbefore, monthafter, is_weekly_mode=False)
        temp = sell_out[sell_out['calendar_yearmonth'].isin(list_period)].copy()
        start = 0
        newnames = ["sellout"+str(x) for x in np.arange(-rangemonth, rangemonth+1)]
        list_to_merge = list()
        for i in sorted(temp.calendar_yearmonth.unique()):
            temp2 = temp[temp['calendar_yearmonth'] == i].copy()
            temp2.rename(columns={'total_volume_hl': newnames[start]}, inplace=True)
            list_to_merge.append(temp2[['customer_planning_group', 'lead_sku', newnames[start]]])
            start += 1
        temp3 = reduce(lambda left, right: pd.merge(left, right, how='outer',
                                                    on=['customer_planning_group', 'lead_sku']), list_to_merge)
        temp3['date_to_predict'] = w
        res = pd.concat([res, temp3])
    return res
