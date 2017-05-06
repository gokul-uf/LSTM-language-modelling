import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
from config import Config as conf
from preprocess import preprocessor
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import LSTMCell

# Input placeholders
data = tf.placeholder(tf.int32, [None, conf.seq_length - 1, 1], "sentences")
next_word = tf.placeholder(tf.int32, [None, conf.seq_length - 1, 1], "next_word")

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
    state = cell.zero_state(conf.batch_size, tf.float32, initializer=xavier_initializer())
    for i in range(conf.seq_length - 1):
        if i > 0:
            scope.reuse_variables()
        lstm_output, state = cell(word_embeddings[:, i, :], state)
        lstm_outputs.append(lstm_output)

# stack the outputs together, reshape, multiply
lstm_outputs = tf.stack(lstm_outputs, axis = 1)
lstm_outputs = tf.reshape(lstm_outputs, [conf.batch_size * (conf.seq_length - 1), conf.num_hidden_state])
assert lstm_outputs.shape == (conf.batch_size * (conf.seq_length - 1), conf.num_hidden_state)
prediction_logits = tf.matmul(lstm_outputs, output_matrix) + output_bias

# reshape the labels
labels = tf.reshape(next_word, [conf.batch_size * (conf.seq_length - 1)])

# Average Cross Entropy loss, compute CE separately to use in testing
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction_logits, labels = labels)
loss = tf.reduce_sum(cross_entropy)

#training
adam = tf.train.AdamOptimizer(conf.lr)
gradients, variables = zip(*adam.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
train_step = adam.apply_gradients(zip(gradients, variables))

# preprocessing
print("Starting preprocessing")
preproc = preprocessor()
preproc.preprocess("../data/sentences.train")

# training
print("Start training")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if not os.path.exists(conf.ckpt_dir):
    os.makedirs(conf.ckpt_dir)
saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    if conf.mode == "TRAIN":
        print("Mode set to TRAIN")
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
            print("Average word-level Loss: {}".format(epoch_loss / (64 * 29 * (len(preproc.lines) / 64))))
            save_path = saver.save(sess, "{}/epoch_{}.ckpt".format(conf.ckpt_dir, i))
            print("Model saved in: {}".format(save_path))

    elif conf.mode == "TEST":
        print("Mode set to TEST")
        if conf.ckpt_file == '':
            print('''conf.ckpt_file is not set,
                set it to the ckpt file in {} folder you want to load'''.format(conf.ckpt_dir))
        print("Loading Model")
        saver.restore(sess, conf.ckpt_dir + conf.ckpt_file)
        for data_batch, label_batch in preproc.get_batch(conf.test_file):
            assert data_batch.shape == (64, 29, 1)
            assert label_batch.shape == (64, 29, 1)
            cross_entropies = sess.run(cross_entropy, feed_dict={data: data_batch, next_word: label_batch})
            cross_entropies = np.asarray(cross_entropies)
            cross_entropies = cross_entropies.reshape(conf.batch_size, (conf.seq_length - 1), conf.vocab_size)
            assert cross_entropies.shape == (conf.batch_size, (conf.seq_length - 1), conf.vocab_size)
            for sentence_id in range(conf.batch_size):
                sentence_cross_entropies = []
                for word_pos in range(conf.seq_length - 1):
                    if preproc.idx2word[data_batch[sentence_id, word_pos, 0]] == '<pad>':
                        break
                    ground_truth_id = label_batch[sentence_id, word_pos, 0]
                    sentence_cross_entropies.append( cross_entropies[ sentence_id, word_pos, ground_truth_id ] )
                sentence_perplexity = 2**( np.mean(sentence_cross_entropies) )
                print(sentence_perplexity)
    else:
        print("ERROR: unknown mode '{}', needs to be 'TRAIN' or 'TEST'".format(conf.mode))
