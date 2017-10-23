import keras
import time

from keras.layers import Dense
from keras.models import Sequential
from pandas import read_csv

from data_utils import get_errors


val_size = 120
test_size = 30
df = read_csv('data/mastigadin.csv', header=None)

y_total = df.iloc[:, -1:].values
x_total = df.iloc[:, :-1].values
y_test = y_total[-test_size:, :]
x_test = x_total[-test_size:, :]
y_train = y_total[:-val_size-test_size, :]
x_train = x_total[:-val_size-test_size, :]
y_val = y_total[-val_size-test_size-1:-test_size, :]
x_val = x_total[-val_size-test_size-1:-test_size, :]

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
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=2)
print('TIME: ' + str(time.time() - tempo))

# predict
y_trained = model.predict(x_train)
y_validated = model.predict(x_val)
y_tested = model.predict(x_test)

# errors
errors_train = get_errors(y_train, y_trained)
print('TRAIN: RMSE - ' + str(errors_train))
errors_val = get_errors(y_val, y_validated)
print('\nVAL:   RMSE - ' + str(errors_val))
errors_test = get_errors(y_test, y_tested)
print('\nTEST:   RMSE - ' + str(errors_test))

# plot
# plot(y_total, y_trained, y_validated, y_tested, margin=.2, tittle='KERAS-'+str(errors_val['rmspe']))
