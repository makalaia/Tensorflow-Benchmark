import tensorflow as tf
import time
import os
import pandas as pd

from sklearn.preprocessing import RobustScaler
from data_utils import get_errors, remove_outliers, shuffle_data, plot
from preprocessing.bcb import BCB
from pandas import read_csv
from preprocessing.feature_selection import FeatureSelection
from math import ceil
from preprocessing.interpret_days import sum_days

seed = 123456
tf.set_random_seed(seed=seed)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

val_size = 60
test_size = 30
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

n_input = x_total.shape[1]
n_outputs = 1
n_hidden = 256

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_outputs])
drop1 = tf.placeholder_with_default(1.0, shape=())
drop2 = tf.placeholder_with_default(1.0, shape=())
drop3 = tf.placeholder_with_default(1.0, shape=())
l2_beta = tf.placeholder_with_default(0.0, shape=())

# Store layers weight & bias
weights = {
    'h1': tf.get_variable('h1', shape=[n_input, n_hidden], initializer=tf.contrib.layers.xavier_initializer()),
    'h2': tf.get_variable('h2', shape=[n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(l2_beta)),
    'h3': tf.get_variable('h3', shape=[n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer()),
    'h4': tf.get_variable('h4', shape=[n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(l2_beta)),
    'h5': tf.get_variable('h5', shape=[n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer()),
    'h6': tf.get_variable('h6', shape=[n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(l2_beta)),
    'h7': tf.get_variable('h7', shape=[n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('out', shape=[n_hidden, n_outputs], initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden])),
    'b2': tf.Variable(tf.zeros([n_hidden])),
    'b3': tf.Variable(tf.zeros([n_hidden])),
    'b4': tf.Variable(tf.zeros([n_hidden])),
    'b5': tf.Variable(tf.zeros([n_hidden])),
    'b6': tf.Variable(tf.zeros([n_hidden])),
    'b7': tf.Variable(tf.zeros([n_hidden])),
    'out': tf.Variable(tf.zeros([n_outputs]))
}

# l2_loss = tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['h4']) + tf.nn.l2_loss(weights['h6'])

# Create model
def mlp(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])), drop1)
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    layer_4 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])), drop2)
    layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5']))
    layer_6 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])), drop3)
    layer_7 = tf.nn.relu(tf.add(tf.matmul(layer_6, weights['h7']), biases['b7']))
    out_layer = tf.matmul(layer_7, weights['out']) + biases['out']
    return out_layer


# Construct model
pred = mlp(X)

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
        avg_cost = 0.
        total_batch = ceil(n_samples / batch_size)
        # Loop over all batches
        if SHUFFLE is True:
            x_trained, y_trained = shuffle_data(x_train, y_train, seed)
        tp = time.time()
        for i in range(total_batch):
            batch_x = x_trained[i * batch_size:(i + 1) * batch_size]
            batch_y = y_trained[i * batch_size:(i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y, drop1: .8, drop2: .7, drop3: .6, l2_beta: .005})
            # Compute average loss
            avg_cost += c / total_batch
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

    plot(y_total, y_trained, y_validated, y_tested, margin=.2, tittle='TF-MAO-' + '-' + str(errors_val['rmspe']))
