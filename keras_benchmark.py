import keras
import time
import numpy as np
import os
import pandas as pd
import random as rn
import tensorflow as tf
import keras.backend as K

from keras.layers import Dense, Dropout, regularizers
from keras.models import Sequential
from pandas import read_csv
from sklearn.preprocessing import RobustScaler

from data_utils import get_errors, plot, remove_outliers
from preprocessing.bcb import BCB
from preprocessing.feature_selection import FeatureSelection
from preprocessing.interpret_days import sum_days

val_size = 120
test_size = 60
dataframe = read_csv('data/daily_data.csv')
dataframe.set_index(list(dataframe)[0], inplace=True)
columns = list(dataframe)

# localização do bloco de produtos
for i in range(-1, -len(columns), -1):
    if 'PD_' in columns[i]:
        init = i + 1
        break

prod = 1
target_product = dataframe.columns[-prod+init]
print(target_product)


df_month = read_csv('data/monthly_data.csv')
df = FeatureSelection().add_prod_delay_correlation(df_month, dataframe.copy(), target_product)

bcb = BCB()
bcb = bcb.get_dataframe(df.index[0], df.index[-1])
if not bcb.empty:
    bcb.set_index(df.index, inplace=True)
    df = pd.concat((df, bcb), axis=1, join='inner')

columns = list(df)
columns[-1], columns[columns.index(target_product)] = columns[columns.index(target_product)], columns[-1]
df = df.reindex(columns=columns)
df.iloc[:, -1:] = remove_outliers(df.iloc[:, -1:])
df = sum_days(df, past_days=31, prevision_days=31)
df.drop('NUM_VENDEDOR', axis=1, inplace=True)

y_total = df.iloc[:, -1:].values
x_total = df.iloc[:, :-1].values
y_test = y_total[-test_size:, :]
x_test = x_total[-test_size:, :]
y_train = y_total[:-val_size-test_size, :]
x_train = x_total[:-val_size-test_size, :]
y_val = y_total[-val_size-test_size-1:-test_size, :]
x_val = x_total[-val_size-test_size-1:-test_size, :]
n_samples = x_train.shape[0]

scalerX = RobustScaler(quantile_range=(10, 90))
scalerY = RobustScaler(quantile_range=(10, 90))
x_train = scalerX.fit_transform(x_train)
y_train = scalerY.fit_transform(y_train)
x_val = scalerX.transform(x_val)
y_val = scalerY.transform(y_val)
x_test = scalerX.transform(x_test)
y_test = scalerY.transform(y_test)

tempo = time.time()
epochs = 200
learning_rate = 0.001
batch_size = 32

# random seed
os.environ['PYTHONHASHSEED'] = '0'
seed = 123456
if seed is not None:
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)
    K.set_session(tf.Session(graph=tf.get_default_graph()))

# Neural net
epochs = 200
optmizer = keras.optimizers.Adam()
model = Sequential()
model.add(Dense(256, input_shape=(x_train.shape[1],)))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(.005)))
model.add(Dropout(.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(.005)))
model.add(Dropout(.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(.005)))
model.add(Dropout(.4))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))

# fit
model.compile(loss='mean_squared_error', optimizer=optmizer)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=2)
print('TIME: ' + str(time.time() - tempo))

# predict
y_trained = model.predict(x_train)
y_validated = model.predict(x_val)
y_tested = model.predict(x_test)

# invert
y_train = scalerY.inverse_transform(y_train)
y_val = scalerY.inverse_transform(y_val)
y_test = scalerY.inverse_transform(y_test)
y_trained = scalerY.inverse_transform(y_trained)
y_validated = scalerY.inverse_transform(y_validated)
y_tested = scalerY.inverse_transform(y_tested)

# errors
print('PRODUTO: ' + columns[-1])
errors_train = get_errors(y_train, y_trained)
print('TRAIN:\nRMSE: ' + str(errors_train['rmse']))
print('RMSPE: ' + str(errors_train['rmspe']))

errors_val = get_errors(y_val, y_validated)
print('\nVAL:\nRMSE: ' + str(errors_val['rmse']))
print('RMSPE: ' + str(errors_val['rmspe']))

errors_test = get_errors(y_test, y_tested)
print('\nTEST:\nRMSE: ' + str(errors_test['rmse']))
print('RMSPE: ' + str(errors_test['rmspe']))

plot(y_total, y_trained, y_validated, y_tested, margin=.2, tittle='TF-MLP-' + '-' + str(errors_val['rmspe']))
