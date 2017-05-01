import numpy as np
import tensorflow as tf
from tqdm import tqdm
from config import Config as conf
from preprocess import preprocessor
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import LSTMCell

'''
TODO:
1. saving the model DONE
2. calculate perplexity
3. load custom embeddings
'''

# Input placeholders
data = tf.placeholder(tf.int32, [conf.batch_size, conf.seq_length - 1, 1], "sentences")
next_word = tf.placeholder(tf.int32, [conf.batch_size, conf.seq_length - 1, 1], "next_word")

# LSTM Matrices
embedding_matrix = tf.get_variable("embed", [conf.vocab_size, conf.embed_size], 
                    tf.float32, initializer=xavier_initializer())
output_matrix = tf.get_variable("output", [conf.num_hidden_state, conf.vocab_size], 
                    tf.float32, initializer=xavier_initializer())
output_bias = tf.get_variable("bias", [conf.vocab_size], 
                    tf.float32, initializer=xavier_initializer())

# embedding lookup
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, data) # shape: (64, 29, 1, 100)
word_embeddings = tf.reshape(word_embeddings, [conf.batch_size, conf.seq_length -1, conf.embed_size]) #shape: (64, 29, 100)
assert word_embeddings.shape == (conf.batch_size, conf.seq_length - 1, conf.embed_size)

# RNN unrolling
print("creating RNN")
lstm_outputs = []
with tf.variable_scope("rnn") as scope:
    cell = LSTMCell(conf.num_hidden_state)
    state = cell.zero_state(conf.batch_size, tf.float32)
    for i in range(conf.seq_length - 1):
        if i > 0:
            scope.reuse_variables()
        lstm_output, state = cell(word_embeddings[:, i, :], state)
        lstm_outputs.append(lstm_output)

# stack the outputs together, reshape, multiply
lstm_outputs = tf.stack(lstm_outputs, axis = 1)
lstm_outputs = tf.reshape(lstm_outputs, [conf.batch_size * (conf.seq_length - 1), conf.num_hidden_state])
assert lstm_outputs.shape == (conf.batch_size * (conf.seq_length - 1), conf.num_hidden_state)
predictions = tf.matmul(lstm_outputs, output_matrix) + output_bias

# reshape the labels
labels = tf.reshape(next_word, [conf.batch_size * (conf.seq_length - 1)])

# Average Cross Entropy loss
loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits = predictions, labels = labels))

#training
adam = tf.train.AdamOptimizer(conf.lr)
gradients, variables = zip(*adam.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
train_step = adam.apply_gradients(zip(gradients, variables))

# preprocessing
print("Starting preprocessing")
preproc = preprocessor()
preproc.preprocess("data/sentences.train")

# training
print("Start training")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if not os.path.exists(conf.ckpt_dir):
    os.makedirs(conf.ckpt_dir)
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(conf.num_epochs):
        epoch_loss = 0
        print("epoch {}".format(i))
        for data_batch, label_batch in tqdm(preproc.get_batch(), total = len(preproc.lines) / 64):
           assert data_batch.shape == (64, 29, 1)
           assert label_batch.shape == (64, 29, 1)
           _, curr_loss = sess.run([train_step, loss], feed_dict = 
                               {data: data_batch, next_word: label_batch})
           epoch_loss += curr_loss
        print("Average Loss: {}".format(epoch_loss / (len(preproc.lines) * 64)))
        save_path = saver.save(sess, "{}/epoch_{}.ckpt".format(conf.ckpt_dir, i+1))
        print("Model saved in: {}".format(save_path))