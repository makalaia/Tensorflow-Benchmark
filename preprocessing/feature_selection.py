import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from data_utils import cross_correlation

np.seterr(divide='ignore', invalid='ignore')


class FeatureSelection(object):
    @staticmethod
    def add_prod_delay_correlation(df_month, dataframe, target, v_shift=1, correlation=.8):
        """
        Realiza consulta mensal dos produtos e ve quais tem correlação com o produto target e add o mesmo no dataframe.
        :param df_month:
        :param dataframe:
        :param target:
        :param v_shift:
        :param correlation:
        :return: Dataframe
        """
        correlation_pearsonr = FeatureSelection.delay_correlation_stats(df_month, target, "pearsonr", v_shift, correlation)
        correlation_spearmanr = FeatureSelection.delay_correlation_stats(df_month, target, "spearmanr", v_shift, correlation)

        # remove produtos repetidos
        for value in correlation_spearmanr[0]:
            if value in correlation_pearsonr[0]:
                correlation_pearsonr[0].remove(value)
        correlation = correlation_pearsonr[0] + correlation_spearmanr[0]
        print(correlation)
        print("Qtd de atributos na entrada: " + str(len(correlation)))
        for column in dataframe.columns:
            if "PD_" in column and column != target and column not in correlation:
                dataframe = dataframe.drop(column, axis=1)

        return dataframe

    @staticmethod
    def delay_correlation_stats(dataframe, target, stats="pearsonr", v_shift=1, correlation=0.8):
        if stats == "pearsonr":
            stats = pearsonr
        else:
            stats = spearmanr

        # realiza o delay e a correlação dos dados do mês
        best_prod = best_value = 0
        prods_correlation = prods_non_correlation = ""
        for column in dataframe.columns:
            if "PD_" in column and column != target:
                tam = dataframe.shape[0] - v_shift
                value = stats(dataframe[target].tail(tam), dataframe[column].shift(v_shift).tail(tam))
                if abs(value[0]) > correlation:
                    prods_correlation = prods_correlation + column + ";"
                else:
                    prods_non_correlation = prods_non_correlation + column + ";"
                if abs(value[0]) > best_value:
                    best_value = abs(value[0])
                    best_prod = column
        prods_correlation = prods_correlation[:-1].split(";")
        prods_non_correlation = prods_non_correlation[:-1].split(";")

        if prods_correlation[0] is "":
            prods_correlation[0] = best_prod
            prods_non_correlation.remove(best_prod)

        return prods_correlation, prods_non_correlation

    @staticmethod
    def cross_correlation(dataframe, target, min_lags, max_lags):
        columns = dataframe.columns
        arg_max = 0
        df = pd.DataFrame(index=dataframe.index)
        for column in columns:
            x, corr = cross_correlation(dataframe[column], dataframe[target], maxlags=max_lags)
            corr = corr[x > min_lags]
            x = x[x > min_lags]
            a_max = np.argmax(abs(corr))
            if a_max > arg_max:
                arg_max = a_max

            # monta novo dataframe
            df[column] = dataframe[column].shift(-a_max)

        return arg_max, df
