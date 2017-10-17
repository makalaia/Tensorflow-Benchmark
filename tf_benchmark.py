import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from pandas import read_csv
from math import ceil

from data_utils import shuffle_data, calculate_rmse, plot, calculate_rmspe, get_errors

val_size = 120
test_size = 30
df = read_csv('data/mastigadin.csv', header=None)

y_total = df.iloc[:, -1:].values
x_total = df.iloc[:, :-1].values
y_test = y_total[-test_size:, :]
x_test = x_total[-test_size:, :]
y_train = y_total[:-val_size - test_size, :]
x_train = x_total[:-val_size - test_size, :]
y_val = y_total[-val_size - test_size - 1:-test_size, :]
x_val = x_total[-val_size - test_size - 1:-test_size, :]
n_samples = x_train.shape[0]

tempo = time.time()
epochs = 200
batch_size = 128

n_input = x_total.shape[1]
n_output = 1
n_hidden = 256

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

# Store layers weight & bias
weight_initializer = tf.contrib.layers.xavier_initializer()
weights = {
    'h1': tf.get_variable('h1', shape=[n_input, n_hidden], initializer=tf.contrib.layers.xavier_initializer()),
    'h2': tf.get_variable('h2', shape=[n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer()),
    'h3': tf.get_variable('h3', shape=[n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer()),
    'h4': tf.get_variable('h4', shape=[n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer()),
    'h5': tf.get_variable('h5', shape=[n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer()),
    'h6': tf.get_variable('h6', shape=[n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer()),
    'h7': tf.get_variable('h7', shape=[n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('out', shape=[n_hidden, n_output], initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden])),
    'b2': tf.Variable(tf.zeros([n_hidden])),
    'b3': tf.Variable(tf.zeros([n_hidden])),
    'b4': tf.Variable(tf.zeros([n_hidden])),
    'b5': tf.Variable(tf.zeros([n_hidden])),
    'b6': tf.Variable(tf.zeros([n_hidden])),
    'b7': tf.Variable(tf.zeros([n_hidden])),
    'out': tf.Variable(tf.zeros([n_output]))
}


# Create model
def multilayer_perceptron(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))
    layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5']))
    layer_6 = tf.nn.relu(tf.add(tf.matmul(layer_5, weights['h6']), biases['b6']))
    layer_7 = tf.nn.relu(tf.add(tf.matmul(layer_6, weights['h7']), biases['b7']))
    out_layer = tf.matmul(layer_7, weights['out']) + biases['out']
    return out_layer


# Construct model
pred = multilayer_perceptron(X)

# Define loss and optimizer
cost = tf.reduce_mean(tf.squared_difference(pred, Y))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

display_step = 1
SHUFFLE = True
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    x_trained, y_trained = shuffle_data(x_train, y_train)
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = ceil(n_samples / batch_size)
        # Loop over all batches
        tp = time.time()
        if SHUFFLE is True:
            x_trained, y_trained = shuffle_data(x_train, y_train)
        for i in range(total_batch):
            batch_x = x_trained[i * batch_size:(i + 1) * batch_size]
            batch_y = y_trained[i * batch_size:(i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost), "TIME: %.2f" % (time.time() - tp))
    print('TIME: ' + str(time.time() - tempo))

    # Test model
    y_trained = sess.run(pred, feed_dict={X: x_train})
    y_validated = sess.run(pred, feed_dict={X: x_val})
    y_tested = sess.run(pred, feed_dict={X: x_test})

# errors
errors_train = get_errors(y_train, y_trained)
print('TRAIN: RMSE - ' + str(errors_train))
errors_val = get_errors(y_val, y_validated)
print('\nVAL:   RMSE - ' + str(errors_val))
errors_test = get_errors(y_test, y_tested)
print('\nTEST:   RMSE - ' + str(errors_test))

# plot
plot(y_total, y_trained, y_validated, y_tested, margin=.2, tittle='TF-MAO-'+str(errors_val['rmspe']))