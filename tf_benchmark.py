import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
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
n_samples = x_train.shape[0]

tempo = time.time()
epochs = 200
batch_size = 64

n_input = 36
n_output = 1
n_hidden = 256

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

# Store layers weight & bias
weights = {
    'h1': tf.get_variable('h1', shape=[n_input, n_hidden]),
    'h2': tf.get_variable('h2', shape=[n_hidden, n_hidden]),
    'h3': tf.get_variable('h3', shape=[n_hidden, n_hidden]),
    'h4': tf.get_variable('h4', shape=[n_hidden, n_hidden]),
    'h5': tf.get_variable('h5', shape=[n_hidden, n_hidden]),
    'h6': tf.get_variable('h6', shape=[n_hidden, n_hidden]),
    'h7': tf.get_variable('h7', shape=[n_hidden, n_hidden]),
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden])),
    'b2': tf.Variable(tf.random_normal([n_hidden])),
    'b3': tf.Variable(tf.random_normal([n_hidden])),
    'b4': tf.Variable(tf.random_normal([n_hidden])),
    'b5': tf.Variable(tf.random_normal([n_hidden])),
    'b6': tf.Variable(tf.random_normal([n_hidden])),
    'b7': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_output]))
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
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        tp = time.time()
        for i in range(total_batch):
            batch_x = x_train[i * batch_size:(i + 1) * batch_size]
            batch_y = y_train[i * batch_size:(i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, cost], feed_dict={X: batch_x, Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost), "TIME: %.2f" % (time.time() - tp))
    print("Optimization Finished!")
    print('TIME: ' + str(time.time() - tempo))

    # Test model
    y_trained = sess.run(pred, feed_dict={X: x_train})
    y_tested = sess.run(pred, feed_dict={X: x_test})

    error_train = calculate_rmse(y_train, y_trained)
    print('TRAIN: RMSE - ' + str(error_train))
    error_test = calculate_rmse(y_test, y_tested)
    print('\nVAL:   RMSE - ' + str(error_test))

plt.plot(y_total, label='REAL DATA')
plt.plot(y_trained, label='TRAINED DATA')
plt.plot(range(len(y_train), len(y_total)), y_tested, label='TEST DATA')
plt.legend()
plt.show()
