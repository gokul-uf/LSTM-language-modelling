import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
from config import Config as conf
from preprocess import preprocessor
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import LSTMCell
import datetime

'''
TODO:
1. saving the model DONE
2. calculate perplexity 
3. load custom embeddings
'''

# Input placeholders
data = tf.placeholder(tf.int32, [None, conf.seq_length - 1, 1], "sentences")
next_word = tf.placeholder(tf.int32, [None, conf.seq_length - 1, 1], "next_word")

# LSTM Matrices
embedding_matrix = tf.get_variable("embed", [conf.vocab_size, conf.embed_size], 
                    tf.float32, initializer=xavier_initializer())
output_matrix = tf.get_variable("output", [conf.proj_hidden_state, conf.vocab_size],
                    tf.float32, initializer=xavier_initializer())
output_bias = tf.get_variable("bias", [conf.vocab_size], 
                    tf.float32, initializer=xavier_initializer())
projection_matrix = tf.get_variable("projection", [conf.num_hidden_state, conf.proj_hidden_state],
                    tf.float32, initializer=xavier_initializer())

# embedding lookup
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, data) # shape: (64, 29, 1, 100)
word_embeddings = tf.reshape(word_embeddings, [conf.batch_size, conf.seq_length -1, conf.embed_size]) #shape: (64, 29, 100)
assert word_embeddings.shape == (conf.batch_size, conf.seq_length - 1, conf.embed_size)

# RNN unrolling
print("creating RNN")
lstm_outputs = []
with tf.variable_scope("rnn") as scope:
    cell = LSTMCell(conf.num_hidden_state, initializer=xavier_initializer())
    state = cell.zero_state(conf.batch_size, tf.float32)
    for i in range(conf.seq_length - 1):
        if i > 0:
            scope.reuse_variables()
        lstm_output, state = cell(word_embeddings[:, i, :], state)
        lstm_outputs.append(lstm_output)

# stack the outputs together, reshape, project, multiply
lstm_outputs = tf.stack(lstm_outputs, axis = 1)
lstm_outputs = tf.reshape(lstm_outputs, [conf.batch_size * (conf.seq_length - 1), conf.num_hidden_state])
assert lstm_outputs.shape == (conf.batch_size * (conf.seq_length - 1), conf.num_hidden_state)
lstm_outputs = tf.matmul(lstm_outputs, projection_matrix)
predictions = tf.matmul(lstm_outputs, output_matrix) + output_bias

# reshape the labels
labels = tf.reshape(next_word, [conf.batch_size * (conf.seq_length - 1)])

# Average Cross Entropy loss, compute CE separately to use in testing
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = predictions, labels = labels)
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
        sentence_perplexities = []
        for data_batch, label_batch in preproc.get_batch(conf.test_file):
           assert data_batch.shape == (64, 29, 1)
           assert label_batch.shape == (64, 29, 1)
           cross_entropies = sess.run(cross_entropy, feed_dict={data: data_batch, next_word: label_batch})
           cross_entropies = np.asarray(cross_entropies)
           cross_entropies = cross_entropies.reshape(conf.batch_size, (conf.seq_length - 1))
           assert cross_entropies.shape == (conf.batch_size, (conf.seq_length - 1))
           for sentence_id in range(conf.batch_size):
               sentence_cross_entropies = [0]  # initial <bos> has likelihood 1
               for word_pos in range(conf.seq_length - 1):
                   if preproc.idx2word[label_batch[sentence_id, word_pos, 0]] == '<pad>':
                       break
                   sentence_cross_entropies.append(cross_entropies[sentence_id, word_pos])
               sentence_perplexity = 2 ** (np.mean(sentence_cross_entropies))
               # print(sentence_perplexity)
               sentence_perplexities.append(sentence_perplexity)
        if not os.path.exists(conf.test_dir):
            os.makedirs(conf.test_dir)
        with open(conf.test_dir + 'sentences_test_perplexity_' + str(datetime.datetime.now()), 'w') as perplexity_file:
            for sentence_id in range(preproc.get_num_lines(conf.test_file)):
                perplexity_file.write(str(sentence_perplexities[sentence_id]))
                perplexity_file.write('\n')
    elif conf.mode == "CONTINUATION":
        print("Mode to greedy sentence CONTINUATION")
        print("Loading Model")
        saver.restore(sess, conf.ckpt_dir + conf.ckpt_file)
        sentence_continuations = []
        for data_batch, label_batch in tqdm(preproc.get_batch(conf.continuation_file), total=preproc.get_num_lines(conf.continuation_file)//conf.batch_size):

            # Store continuation position in label_batch
            batch_id2num_words = np.ndarray(shape=conf.batch_size,dtype=int)

            # Compute continuation position in label_batch by finding position of '<eos>' in label_batch for each sentence
            sentence_complete_batch = np.ndarray((conf.batch_size),dtype=bool)
            sentence_complete_batch.fill( True )
            for batch_id in range(conf.batch_size):
                batch_id2num_words[batch_id] = conf.seq_length-1
                for word_pos in range(conf.completed_sentence_length-2):
                    if preproc.idx2word[ label_batch[batch_id, word_pos, 0] ] == '<eos>':
                        batch_id2num_words[batch_id] = word_pos
                        sentence_complete_batch[batch_id] = False

                        # sentence = []
                        # for i in range(word_pos+1):
                        #     sentence.append(preproc.idx2word[label_batch[batch_id, i, 0]])
                        # print("Found sentence: " + ' '.join(sentence))

                        break
                if sentence_complete_batch[batch_id]: # sentence already fills 20 symbols including '<bos>' and '<eos>'
                    print("Found a sentence that is already complete: " + ' '.join(preproc.idx2word[label_batch[ batch_id, :, 0]]))
                    data_batch[batch_id, conf.completed_sentence_length-1, 0] = preproc.word2idx['<eos>']
                    label_batch[batch_id, conf.completed_sentence_length-2, 0] = preproc.word2idx['<eos>']

            #print("Found sentence endings at..." + str(batch_id2num_words))

            # Run session until all sentences have at least 18 words (+ <box> + <eos> gives 20)
            while np.logical_and(batch_id2num_words < conf.completed_sentence_length-2,
                                 np.logical_not(sentence_complete_batch)).any():
                # import pdb; pdb.set_trace()
                #print("Sentence endings at..." + str(batch_id2num_words))
                prediction_argmax_logits = np.argmax( np.reshape( sess.run(predictions, feed_dict={data: data_batch, next_word: label_batch}),
                                                                  [conf.batch_size, (conf.seq_length - 1), conf.vocab_size] ), axis=2)
                for batch_id in range(conf.batch_size):
                    if batch_id2num_words[batch_id] < conf.completed_sentence_length-2 and not sentence_complete_batch[batch_id]: #label_batch[batch_id, batch_id2num_words[batch_id], 0] != preproc.word2idx['<eos>']:
                        if preproc.idx2word[ prediction_argmax_logits[batch_id, batch_id2num_words[batch_id]] ] not in {'<bos>','<pad>'}: # <unk> is acceptable
                            data_batch[ batch_id, batch_id2num_words[batch_id]+1, 0] = prediction_argmax_logits[batch_id, batch_id2num_words[batch_id]]
                            label_batch[batch_id, batch_id2num_words[batch_id],   0] = prediction_argmax_logits[batch_id, batch_id2num_words[batch_id]]
                            batch_id2num_words[batch_id] += 1
                            #print("Updating " + str(batch_id) + ": " + ' '.join([preproc.idx2word[ label_batch[batch_id, i,   0] ] for i in range(conf.completed_sentence_length)]) )
                            if prediction_argmax_logits[batch_id, batch_id2num_words[batch_id]-1] == preproc.word2idx['<eos>']:
                                sentence_complete_batch[batch_id] = True
                            elif batch_id2num_words[batch_id] >= conf.completed_sentence_length-2: # The completed sentence must not be longer than 20 symbols, append <eos> at the end
                                data_batch[batch_id, batch_id2num_words[batch_id] + 1, 0] = preproc.word2idx['<eos>']
                                label_batch[batch_id, batch_id2num_words[batch_id], 0] = preproc.word2idx['<eos>']
                                batch_id2num_words[batch_id] += 1
                                sentence_complete_batch[batch_id] = True

            for batch_id in range(conf.batch_size):
                completed_sentence = []
                for word_pos in range(conf.completed_sentence_length-1):
                    word = preproc.idx2word[label_batch[batch_id, word_pos, 0]]
                    completed_sentence.append( word )
                    if word == '<eos>':
                        print(' '.join(completed_sentence))
                        sentence_continuations.append(' '.join(completed_sentence))
                        break
        if not os.path.exists(conf.continuation_dir):
            os.makedirs(conf.continuation_dir)
        with open(conf.continuation_dir + 'sentences.continuation_' + str(datetime.datetime.now()), 'w') as continuation_file:
            for sentence_id in range(preproc.get_num_lines(conf.continuation_file)):
                continuation_file.write(str(sentence_continuations[sentence_id]))
                continuation_file.write('\n')
    else:
        print("ERROR: unknown mode '{}', needs to be 'TRAIN' or 'TEST' or 'CONTINUATION'".format(conf.mode))
