import time

import numpy as np
import pandas as pd
import quandl
from dateutil.relativedelta import relativedelta

from preprocessing import indice_bcb

INDEX_MENSAL = ['ICC', 'DESEMPREGO', 'VENDAS_AUTOPECAS_SUDESTE', 'PIB', 'PRODUCAO_ACO', 'IPCA', 'VENDAS_MOTOCICLOS', 'VENDAS_PARTES_PECAS_AUTOMOVEIS']
INDEX_DIARIO = ['DOLAR_VENDA']
INDEX_TOTAL = INDEX_DIARIO + INDEX_MENSAL
BCB_INDEX_DEFAULT = ['ICC', 'DESEMPREGO', 'PIB', 'IPCA', 'DOLAR_VENDA']
with open('keys/quandl.key', 'r') as file:
    QUANDL_API_KEY = file.readline()
DATE_BEGIN = pd.to_datetime('1/1/2013')
DATE_END = pd.to_datetime('today')


class BCB:
    def __init__(self, codparc=None, index=BCB_INDEX_DEFAULT):
        quandl.ApiConfig.api_key = QUANDL_API_KEY
        if index is None:
            self.index_mensal = INDEX_MENSAL
            self.index_diario = INDEX_DIARIO
        else:
            self.index_diario = [i for i in index if i in INDEX_DIARIO]
            self.index_mensal = [i for i in index if i in INDEX_MENSAL]
        self.codparc = str(codparc)
        self.date_begin = DATE_BEGIN
        self.date_end = DATE_END
        self.data_diaria = self.get_diario()
        self.data_mensal = self.to_diario(self.get_mensal())

    def get_mensal(self):
        start = self.date_begin - relativedelta(months=4)
        cod = indice_bcb.list_to_cod(self.index_mensal)
        monthly_array = BCB.get_from_quandl(cod, trim_start=start)
        monthly_array = monthly_array.reindex(pd.date_range(start, end=self.date_end, freq='M'))
        monthly_array.index = pd.date_range(start, periods=monthly_array.shape[0], freq='MS')
        return monthly_array

    def to_diario(self, data_mensal):
        start = self.date_begin - relativedelta(months=4)
        data_mensal = data_mensal.resample('D').asfreq()
        data_mensal = data_mensal.reindex(index=pd.date_range(start=start, end=self.date_end, freq='D'))
        return data_mensal

    def get_diario(self):
        start = self.date_begin - relativedelta(months=4)
        cod = indice_bcb.list_to_cod(self.index_diario)
        daily_array = BCB.get_from_quandl(cod, trim_start=start)
        daily_array = daily_array.reindex(index=pd.date_range(start=start, end=self.date_end, freq='D'))
        return daily_array

    def get_dataframe(self, date_begin=None, date_end=None):
        if date_begin is not None:
            self.date_begin = pd.to_datetime(date_begin, yearfirst=True)
        if date_end is not None:
            self.date_end = pd.to_datetime(date_end, yearfirst=True)

        array = pd.concat((self.data_mensal, self.data_diaria), axis=1)
        array = BCB.interpolate(array)
        array = array.reindex(pd.date_range(end=self.date_end, periods=array.shape[0], freq='D'))
        for i in range(array.shape[1]):
            column = np.asarray(array.iloc[:, i])
            while np.isnan(column[-1]):
                column = np.roll(column, 1)
            array.iloc[:, i] = column
        array = array.loc[self.date_begin:]
        array = array.dropna(axis=1)
        return array

    @staticmethod
    def interpolate(array):
        array = array.interpolate(method='cubic', axis=0)
        return array

    @staticmethod
    def get_from_quandl(index, trim_start, max_retries=10):
        time_to_retry = .5
        data = None
        for i in range(0, max_retries):
            try:
                data = quandl.get(index, trim_start=trim_start)
                break
            except Exception:
                if i + 1 == max_retries:
                    raise TimeoutError('Nao foi possivel receber os dados do quandl em tempo suficiente')
                time.sleep(time_to_retry)
        return data
