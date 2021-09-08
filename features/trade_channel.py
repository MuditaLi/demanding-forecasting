""" Build all trade channel features (on/off-trade) """
import preprocessor.names as n


def build_on_trade_feature(data, cpg_trade_channel):
    """

    """
    cpgs = data[[n.FIELD_CUSTOMER_GROUP]].drop_duplicates()
    cpgs[n.FIELD_IS_ON_TRADE] = 0

    on_trade_cpgs = cpg_trade_channel[
        cpg_trade_channel[n.FIELD_CUSTOMER_TRADE_CHANNEL] == 'On Trade'
    ][n.FIELD_CUSTOMER_GROUP].values

    cpgs[n.FIELD_IS_ON_TRADE] = cpgs[n.FIELD_CUSTOMER_GROUP].isin(on_trade_cpgs).apply(int)
    return cpgs
