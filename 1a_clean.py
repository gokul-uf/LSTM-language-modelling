import tensorflow as tf
import numpy as np
from config import Config as conf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import LSTMCell

data = tf.placeholder(tf.float32, [conf.batch_size, conf.seq_length, 1], "sentences")

embedding_matrix = tf.get_variable("embed", [conf.vocab_size, conf.embed_size], tf.float32, initializer=xavier_initializer())
output_matrix = tf.get_variable("output", [conf.num_hidden_state, conf.vocab_size], tf.float32, initializer=xavier_initializer())
output_bias = tf.get_variable("bias", [conf.vocab_size], tf.float32, initializer=xavier_initializer())

cell = LSTMCell(conf.num_hidden_state)
state = cell.zero_state(conf.batch_size, tf.float32)
Y_pred = []

with tf.variable_scope("myrnn") as scope: # http://stackoverflow.com/questions/36941382/tensorflow-shared-variables-error-with-simple-lstm-network
    for i in range(conf.seq_length):
        if i > 0:
            scope.reuse_variables() 
        lstm_op, state = cell(data[:,i,:], state)
        interim = tf.nn.softmax(tf.matmul(lstm_op, output_matrix) + output_bias)
        Y_pred.append(interim)

Y_pred = tf.stack(Y_pred, axis=1)
Y_pred = tf.reshape(Y_pred, shape = [conf.batch_size, conf.seq_length, 1, conf.vocab_size])

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(data, tf.int32), logits = Y_pred)
print cross_entropy.shape