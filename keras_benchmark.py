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
from trainer import Trainer

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
trainer = Trainer(df_daily=dataframe, df_monthly=df_month)
x_train, y_train, x_val, y_val, x_test, y_test = trainer.load_data(val_size=val_size, test_size=test_size, target_column=target_product)
y_total = np.concatenate((y_train, y_val, y_test))
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
y_train = trainer.inverse_transformY(y_train)
y_val = trainer.inverse_transformY(y_val)
y_test = trainer.inverse_transformY(y_test)
y_trained = trainer.inverse_transformY(y_trained)
y_validated = trainer.inverse_transformY(y_validated)
y_tested = trainer.inverse_transformY(y_tested)

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

plot(y_total, y_trained, y_validated, y_tested, margin=.2, tittle='KERAS-MLP-' + '-' + str(errors_val['rmspe']))
