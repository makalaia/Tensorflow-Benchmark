import tensorflow as tf
import time
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from tensorflow.contrib.rnn import LSTMBlockCell

from data_utils import get_errors, remove_outliers, shuffle_data, plot
from preprocessing.bcb import BCB
from pandas import read_csv
from preprocessing.feature_selection import FeatureSelection
from math import ceil
from preprocessing.interpret_days import sum_days


def reshape_input(x, back):
    x = np.asarray(x)
    reshaped_x = np.zeros((x.shape[0] - back, back, x.shape[1]))
    for i in range(0, x.shape[0] - back):
        reshaped_x[i] = x[i:i + back, :]
    return reshaped_x


def reshape_output(y, back):
    y = np.asarray(y)
    return y[back:]

seed = 123456
tf.set_random_seed(seed=seed)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
y_train = y_total[:-val_size-test_size, :]
x_train = x_total[:-val_size-test_size, :]
n_samples = x_train.shape[0]

# normalize
scalerX = RobustScaler(quantile_range=(10, 90))
scalerX.fit(x_train)
scalerY = RobustScaler(quantile_range=(10, 90))
scalerY.fit(y_train)
yt = scalerY.transform(y_total)
xt = scalerX.transform(x_total)

# reshape input to be [samples, time steps, features]
timesteps = 7
xt = reshape_input(xt, timesteps)
yt = reshape_output(yt, timesteps)
train_size = len(y_total) - val_size - test_size - timesteps
x_train, y_train = xt[:train_size, :], yt[:train_size, :]
x_val, y_val = xt[train_size:-test_size, :], yt[train_size:-test_size, :]
x_test, y_test = xt[-test_size:, :], yt[-test_size:, :]

tempo = time.time()
epochs = 200
learning_rate = 0.001
batch_size = 32

n_input = x_total.shape[1]
n_outputs = 1
n_hidden = 128

# tf Graph input
X = tf.placeholder("float", [None, timesteps, n_input])
Y = tf.placeholder("float", [None, n_outputs])

# Store layers weight & bias
weights = {
    'out': tf.get_variable('out', shape=[n_hidden, n_outputs], initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'out': tf.Variable(tf.zeros([n_outputs]))
}


# Construct model
def RNN(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)
    lstm_cell = LSTMBlockCell(n_hidden, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(X, weights, biases)

# Define loss and optimizer
reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
cost = tf.reduce_mean(tf.squared_difference(pred, Y) + reg_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

display_step = 1
SHUFFLE = True
config = tf.ConfigProto(
        device_count = {'cpu': 0}
        )

with tf.Session(config=config) as sess:
    sess.run(init)
    # Training cycle
    x_trained, y_trained = x_train, y_train
    for epoch in range(epochs):
        total_batch = ceil(n_samples / batch_size)
        # Loop over all batches
        if SHUFFLE is True:
            x_trained, y_trained = shuffle_data(x_train, y_train, seed)
        tp = time.time()
        for i in range(total_batch):
            batch_x = x_trained[i * batch_size:(i + 1) * batch_size]
            batch_y = y_trained[i * batch_size:(i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y})
            # Compute average loss
        # Display logs per epoch step
        if epoch % display_step == 0:
            y_trained = sess.run(pred, feed_dict={X: x_train})
            y_validated = sess.run(pred, feed_dict={X: x_val})
            errors_train = get_errors(y_train, y_trained)
            errors_val = get_errors(y_val, y_validated)
            print("Epoch:", '%d/%d' % ((epoch + 1), epochs), "train_error={:.5f}".format(errors_train['rmse']), "- val_error={:.5f}".format(errors_val['rmse']), "TIME: %.2f" % (time.time() - tp))
    print('TEMPO TOTAL: ' + str(time.time() - tempo))

    # Test model
    y_trained = sess.run(pred, feed_dict={X: x_train})
    y_validated = sess.run(pred, feed_dict={X: x_val})
    y_tested = sess.run(pred, feed_dict={X: x_test})

    # invert
    y_train = scalerY.inverse_transform(y_train)
    y_val = scalerY.inverse_transform(y_val)
    y_test = scalerY.inverse_transform(y_test)
    y_trained = scalerY.inverse_transform(y_trained)
    y_validated = scalerY.inverse_transform(y_validated)
    y_tested = scalerY.inverse_transform(y_tested)

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

    plot(y_total[timesteps:], y_trained, y_validated, y_tested, margin=.2, tittle='TF-LSTM-' + '-' + str(errors_val['rmspe']))