from tensorflow.models.rnn import rnn_cell, rnn
import tensorflow as tf
import numpy as np
import input_data
sess = tf.Session()  

'''
Classify MNIST using LSTM running row by row. 

Good:
* No compilation time at all, which is cool.

Bad:
* Problem is that has all dimensions hard coded, which sucks.

Inspired by:
https://github.com/nlintz/TensorFlow-Tutorials
'''

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
     
def get_lstm(num_steps, input_dim, hidden_dim, output_dim, batch_size):
    # Define input
    input = tf.placeholder("float", [batch_size, num_steps, input_dim])
    desired = tf.placeholder("float", [batch_size, 10])
    # Define parameters
    i2h = init_weights([input_dim, hidden_dim])
    h2o = init_weights([hidden_dim, output_dim])
    bi = init_weights([hidden_dim])
    bo = init_weights([output_dim])
    
    # prepare input
    # input shape: (batches, num_steps, input_dim)
    X2 = tf.transpose(input, [1, 0, 2])  # (num_steps, batch_size, input_dim)
    # tf.reshape does not accept X.get_shape elements as input :(
    X3 = tf.reshape(X2, [num_steps*batch_size, dim]) # (num_steps*batch_size, input_dim)
    # project to hidden state dimension
    X4 = tf.matmul(X3, i2h) + bi # (num_steps*batch_size, hidden_dim)

    # LSTM for loop expects a list as input, here we slice X3 into pieces of (batch_size, hidden_dim)
    # tf.split expects as input a axis to slice, number of slices and a tensor
    Xh = tf.split(0, num_steps, X4)
    
    
    initializer = tf.random_uniform_initializer(-.01, .01)
    # INNER LOOP
    # There are two ways of calculating the inner loop of an RNN
    with tf.variable_scope("RNN", reuse=None, initializer=initializer): # this is necessary
        lstm_cell = rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0)
        initial_state = lstm_cell.zero_state(batch_size, tf.float32)
        # Explicitly calling a for loop inside the scope
        #for time_step, input_ in enumerate(inputs):
        #    if time_step > 0: tf.get_variable_scope().reuse_variables()
        #    (cell_output, state) = lstm_cell(input_, initial_state)
        #    outputs.append(cell_output)
        #    states.append(state)
    
        # or simply using rnn(cell, inputs, initial_state=init_state)
        lstm_outputs, lstm_states = rnn.rnn(lstm_cell, Xh, initial_state=initial_state)
        sess.run(tf.initialize_all_variables()) # it didn't work for me initializing outside the scope
    
    # calculate output
    Y = lstm_outputs[-1] # outputs is a list, we get the last value
    output = tf.matmul(Y, h2o) + bo
    
    return input, output, desired
