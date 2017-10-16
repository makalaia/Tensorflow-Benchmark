import keras
import numpy as np
import time
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential
from pandas import read_csv


def calculate_rmse(real, predict):
    m = len(real)
    return np.sqrt(np.sum(np.power((real - predict), 2)) / m)


test_size = 150
df = read_csv('data/mastigadin.csv', header=None)
df.set_index(list(df)[0], inplace=True)
y_total = df.iloc[:, -1:].values
x_total = df.iloc[:, :-1].values
y_train = y_total[:-test_size, :]
x_train = x_total[:-test_size, :]
y_test = y_total[-test_size:, :]
x_test = x_total[-test_size:, :]

tempo = time.time()
# Neural net
epochs = 200
batch_size = 64
optmizer = keras.optimizers.Adam()
model = Sequential()
model.add(Dense(256, input_shape=(x_train.shape[1],)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))

# fit
model.compile(loss='mean_squared_error', optimizer=optmizer)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=2)
print('TIME: ' + str(time.time() - tempo))

# predict
y_trained = model.predict(x_train)
y_tested = model.predict(x_test)

# errors
error_train = calculate_rmse(y_train, y_trained)
print('TRAIN: RMSE - ' + str(error_train))
error_test = calculate_rmse(y_test, y_tested)
print('\nVAL:   RMSE - ' + str(error_test))

# plot
plt.plot(y_total, label='REAL DATA')
plt.plot(y_trained, label='TRAINED DATA')
plt.plot(range(len(y_train), len(y_total)), y_tested, label='TEST DATA')
plt.legend()
plt.show()
