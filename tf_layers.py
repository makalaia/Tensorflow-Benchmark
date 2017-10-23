import numpy as np
import tensorflow as tf
import time
from pandas import read_csv
from data_utils import get_errors

val_size = 120
test_size = 30
df = read_csv('data/mastigadin2.csv', header=None)

y_total = df.iloc[:, -1:].values
x_total = df.iloc[:, :-1].values
y_test = y_total[-test_size:, :]
x_test = x_total[-test_size:, :]
y_train = y_total[:-val_size - test_size, :]
x_train = x_total[:-val_size - test_size, :]
y_val = y_total[-val_size - test_size - 1:-test_size, :]
x_val = x_total[-val_size - test_size - 1:-test_size, :]
n_samples = x_train.shape[0]

# Parameters
learning_rate = 0.001
batch_size = 64
display_step = 100

# Network Parameters
epochs = 200
n_hidden = 256  # 1st layer number of neurons
num_input = x_total.shape[1]  # MNIST data input (img shape: 28*28)
n_output = 1  # MNIST total classes (0-9 digits)

tempo = time.time()
# Define the neural network
def neural_net(x_dict):
    x = x_dict['input']
    # TF Estimator input is a dict, in case of multiple inputs
    layer_1 = tf.layers.dense(x, n_hidden, activation=tf.nn.relu)
    layer_2 = tf.layers.dense(layer_1, n_hidden, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(.005))
    layer_3 = tf.layers.dense(layer_2, n_hidden, activation=tf.nn.relu)
    layer_4 = tf.layers.dense(layer_3, n_hidden, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(.005))
    layer_5 = tf.layers.dense(layer_4, n_hidden, activation=tf.nn.relu)
    layer_6 = tf.layers.dense(layer_5, n_hidden, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(.005))
    layer_7 = tf.layers.dense(layer_6, n_hidden, activation=tf.nn.relu)
    out_layer = tf.layers.dense(layer_7, n_output)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    pred = neural_net(features)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.squared_difference(pred, labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.mean_squared_error(labels=labels, predictions=pred)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'mse': acc_op})

    return estim_specs


# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'input': x_train}, y=y_train,
    batch_size=batch_size, num_epochs=epochs, shuffle=True)
# Train the Model
model.train(input_fn)
print('TIME: ' + str(time.time() - tempo))

# predicts
# trained
input_fn = tf.estimator.inputs.numpy_input_fn(
           x={'input': x_train}, batch_size=batch_size, shuffle=False)
y_trained = np.ravel(list(model.predict(input_fn)))

# validated
input_fn = tf.estimator.inputs.numpy_input_fn(
           x={'input': x_val}, batch_size=batch_size, shuffle=False)
y_validated = np.ravel(list(model.predict(input_fn)))

# tested
input_fn = tf.estimator.inputs.numpy_input_fn(
           x={'input': x_test}, batch_size=batch_size, shuffle=False)
y_tested = np.ravel(list(model.predict(input_fn)))


# errors
errors_train = get_errors(y_train, y_trained)
print('TRAIN: RMSE - ' + str(errors_train))
errors_val = get_errors(y_val, y_validated)
print('\nVAL:   RMSE - ' + str(errors_val))
errors_test = get_errors(y_test, y_tested)
print('\nTEST:   RMSE - ' + str(errors_test))

# plot(y_total, y_trained, y_validated, y_tested, margin=.2, tittle='TF-LAYERS-' + str(errors_val['rmspe']))