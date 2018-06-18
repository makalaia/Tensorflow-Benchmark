from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data_utils import get_numbers_from_string


def one_hot_months(df, fut_days=30, add_days=0):
    if add_days > fut_days:
        raise ValueError('add_days deve ser <= fut_days')
    months = 12
    mes = df['MES'].values
    index = df.index

    # preloc meses
    column_name_month = list()
    for i in range(0, months):
        column_name_month.append('MES_' + str(i + 1))
    meses = np.zeros((df.shape[0], months))

    # increment mes
    append = np.zeros(add_days, dtype=int)
    ind = np.zeros(add_days, dtype=str)
    for i in range(0, add_days):
        tempo = datetime.strptime(df.index.values[-1], "%Y-%m-%d").date() + timedelta(i + 1)
        append[i] = int(tempo.strftime("%m"))
        ind[i] = tempo.strftime("%Y-%m-%d")
    mes = np.append(mes, append)

    for i in range(0, df.shape[0] - fut_days + add_days):
        # meses
        for p in range(1, fut_days + 1):
            for j in range(0, months):
                if mes[i + p] == j + 1:
                    meses[i, j] = meses[i, j] + 1
                    break
    meses = pd.DataFrame(meses, index=index, columns=column_name_month)
    if fut_days != add_days:
        meses = meses.iloc[:-fut_days + add_days, :]
    return meses


def one_hot_weeks(df, fut_days=30, add_days=0):
    if add_days > fut_days:
        raise ValueError('add_days deve ser <= fut_days')
    weekdays = 7
    dia_da_semana = df['DIA_SEMANA'].values
    index = df.index

    # preloc dias da semana
    column_name_week = list()
    for i in range(0, weekdays):
        column_name_week.append('DIA_SEMANA_' + str(i + 1))
    dias_da_semana = np.zeros((df.shape[0], weekdays))

    # increment semana
    append = np.zeros(add_days, dtype=int)
    s = dia_da_semana[-1]
    for i in range(0, add_days):
        append[i] = 1 + (s + i) % 7
    dia_da_semana = np.append(dia_da_semana, append)

    # hot enconde dias da semana
    for i in range(0, df.shape[0] - fut_days + add_days):
        for p in range(0, fut_days):
            for s in range(0, weekdays):
                if dia_da_semana[i + p] == s + 1:
                    dias_da_semana[i, s] = dias_da_semana[i, s] + 1
                    break
    dias_da_semana = pd.DataFrame(dias_da_semana, index=index, columns=column_name_week)
    if fut_days != add_days:
        dias_da_semana = dias_da_semana.iloc[:-fut_days + add_days, :]
    return dias_da_semana


def sum_past_days(df, past_days=30, fut_days=None, one_hot_w=True, one_hot_m=True):
    if fut_days is None:
        fut_days = past_days
    product_names = list()
    columns = list(df)
    index = df.iloc[past_days:, :].index
    try:
        df.iloc[:, 0].values.astype('float64')
    except ValueError:
        df.drop(df.columns[0], axis=1, inplace=True)

    num_products = 0
    for i in columns:
        if 'PD_' in i:
            product_names.append(i)
            num_products += 1

    resto = list()
    for i in columns:
        if 'PD_' not in i and 'MES' not in i and 'DIA_SEMANA' not in i:
            resto.append(i)
    vendas = df[product_names].values
    if one_hot_m:
        meses = one_hot_months(df, fut_days=fut_days, add_days=fut_days)
    else:
        meses = df['MES']
    if one_hot_w:
        dias_da_semana = one_hot_weeks(df, fut_days=fut_days, add_days=fut_days)
    else:
        dias_da_semana = df['DIA_SEMANA']

    # produtos
    column_name_product = list()
    for i in product_names:
        column_name_product.append(str(i))
    produtos = np.zeros((len(vendas), len(column_name_product)))
    for i in range(past_days, len(vendas)):
        for j in range(0, num_products):
            produtos[i, j] = np.sum(vendas[i - past_days:i, j])
    produtos = produtos[past_days:]
    produtos = pd.DataFrame(produtos, index=index, columns=column_name_product)

    # montar dataframe
    dataframe = df[resto].iloc[past_days:].join(list((meses, dias_da_semana, produtos)))
    return dataframe


def sum_days(df, past_days=30, prevision_days=30, num_products=None, one_hot_w=True, one_hot_m=True):
    product_names = list()
    columns = list(df)
    index = df.iloc[past_days:-prevision_days, :].index
    try:
        df.values.astype('float64')
    except ValueError:
        df.drop(df.columns[0], axis=1, inplace=True)
    if num_products is None:
        num_products = 0
        for i in columns:
            if 'PD_' in i:
                product_names.append(i)
                num_products += 1

    resto = list()
    for i in columns:
        if 'PD_' not in i and 'MES' not in i and 'DIA_SEMANA' not in i:
            resto.append(i)
    vendas = df[product_names].values
    if one_hot_m:
        meses = one_hot_months(df, prevision_days).iloc[past_days:, :]
    else:
        meses = df['MES']
    if one_hot_w:
        dias_da_semana = one_hot_weeks(df, prevision_days).iloc[past_days:, :]
    else:
        dias_da_semana = df['DIA_SEMANA']



    # produtos
    column_name_product = list()
    for i in product_names:
        column_name_product.append(str(i))
    produtos = np.zeros((len(vendas), len(column_name_product)))
    for i in range(past_days, len(vendas) - prevision_days):
        for j in range(0, num_products):
            produtos[i, j] = np.sum(vendas[i - past_days:i, j])
    produtos = produtos[past_days:-prevision_days]
    produtos = pd.DataFrame(produtos, index=index, columns=column_name_product)

    # output
    output = np.zeros((len(vendas), 1))
    for i in range(prevision_days, len(vendas) - prevision_days + 1):
        output[i - 1, -1] = np.sum(vendas[i:i + prevision_days, -1])
    output = output[past_days:-prevision_days]
    produto_analisado = 'PRODUTO_' + str(get_numbers_from_string(product_names[-1]))
    output = pd.DataFrame(output, index=index, columns=[produto_analisado])

    # montar dataframe
    dataframe = df[resto].iloc[past_days:-prevision_days].join(list((meses, dias_da_semana, produtos, output)))
    return dataframe


def sum_days_csv(input_path, past_days=30, prevision_days=30):
    dataframe = pd.read_csv(input_path)
    dataframe = sum_days(dataframe, past_days=past_days, prevision_days=prevision_days)
    return dataframe


def __calculate_days_gone__(vendas, index, past_days, prevision_days, days_gone):
    output = np.zeros((len(vendas), 1))
    for i in range(prevision_days, len(vendas) - prevision_days + 1):
        output[i - 1] = np.sum(vendas[i:i + days_gone])
    output = output[past_days:-prevision_days, :]
    output = pd.DataFrame(output, index=index, columns=['FUT' + str(days_gone)])
    return output


def sum_days_gone(df, past_days=30, prevision_days=30, days_gone=20):
    product_names = list()
    columns = list(df)
    index = df.iloc[past_days:-prevision_days, :].index
    try:
        df.values.astype('float64')
    except ValueError:
        df.drop(df.columns[0], axis=1, inplace=True)
    num_products = 0
    for i in columns:
        if 'PD_' in i:
            product_names.append(i)
            num_products += 1

    resto = list()
    for i in columns:
        if 'PD_' not in i and 'MES' not in i and 'DIA_SEMANA' not in i:
            resto.append(i)
    vendas = df[product_names].values
    meses = one_hot_months(df, fut_days=prevision_days)
    dias_da_semana = one_hot_weeks(df, fut_days=prevision_days)
    # produtos
    column_name_product = list()
    for i in product_names:
        column_name_product.append(str(i))
    produtos = np.zeros((len(vendas), len(column_name_product)))
    for i in range(past_days, len(vendas) - prevision_days):
        for j in range(0, num_products):
            produtos[i, j] = np.sum(vendas[i - past_days:i, j])
    produtos = produtos[past_days:-prevision_days]
    produtos = pd.DataFrame(produtos, index=index, columns=column_name_product)

    # output
    output = np.zeros((len(vendas), 2))
    for i in range(prevision_days, len(vendas) - prevision_days + 1):
        output[i-1, -1] = np.sum(vendas[i:i + prevision_days, -1])
        output[i-1, -2] = np.sum(vendas[i:i + days_gone, -1])
    output = output[past_days:-prevision_days, :]
    produto_analisado = 'PRODUTO_' + str(get_numbers_from_string(product_names[-1]))
    output = pd.DataFrame(output, index=index, columns=['FUT20', produto_analisado])

    # montar dataframe
    dataframe = df[resto].iloc[past_days:-prevision_days].join(list((meses, dias_da_semana, produtos, output)))
    return dataframe
