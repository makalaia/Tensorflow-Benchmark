from abc import abstractmethod, ABCMeta

import pandas as pd

from sklearn.preprocessing import RobustScaler
from data_utils import remove_outliers
from preprocessing.bcb import BCB
from preprocessing.feature_selection import FeatureSelection
from preprocessing.interpret_days import sum_days


class Trainer(metaclass=ABCMeta):
    def __init__(self, df_daily, df_monthly):
        self.df_daily = df_daily
        self.df_monthly = df_monthly
        self.scalerX = RobustScaler(quantile_range=(10, 90))
        self.scalerY = RobustScaler(quantile_range=(10, 90))
    
    def load_data(self, val_size, test_size, target_column):
        df = FeatureSelection().add_prod_delay_correlation(dataframe=self.df_daily, df_month=self.df_monthly.copy(), target=target_column)

        bcb = BCB()
        bcb = bcb.get_dataframe(df.index[0], df.index[-1])
        if not bcb.empty:
            bcb.set_index(df.index, inplace=True)
            df = pd.concat((df, bcb), axis=1, join='inner')

        columns = list(df)
        columns[-1], columns[columns.index(target_column)] = columns[columns.index(target_column)], columns[-1]
        df = df.reindex(columns=columns)
        df.iloc[:, -1:] = remove_outliers(df.iloc[:, -1:])
        df = sum_days(df, past_days=31, prevision_days=31)
        df.drop('NUM_VENDEDOR', axis=1, inplace=True)

        y_total = df.iloc[:, -1:].values
        x_total = df.iloc[:, :-1].values
        y_test = y_total[-test_size:, :]
        x_test = x_total[-test_size:, :]
        y_train = y_total[:-val_size - test_size, :]
        x_train = x_total[:-val_size - test_size, :]
        y_val = y_total[-val_size - test_size - 1:-test_size, :]
        x_val = x_total[-val_size - test_size - 1:-test_size, :]

        x_train = self.scalerX.fit_transform(x_train)
        y_train = self.scalerY.fit_transform(y_train)
        x_val = self.scalerX.transform(x_val)
        y_val = self.scalerY.transform(y_val)
        x_test = self.scalerX.transform(x_test)
        y_test = self.scalerY.transform(y_test)
        return x_train, y_train, x_val, y_val, x_test, y_test

    # @abstractmethod
    # def train(self):
    #     pass
    #
    # @abstractmethod
    # def predict(self):
    #     pass

    def inverse_transformX(self, df):
        return self.scalerX.inverse_transform(df)

    def inverse_transformY(self, df):
        return self.scalerY.inverse_transform(df)
