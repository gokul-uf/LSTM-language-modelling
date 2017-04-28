# The dimensions for data are [Batch Size, Sequence Length, Input Dimension]. 
'''
For each LSTM cell that we initialise, we need to supply a value for the hidden dimension, 
or as some people like to call it, the number of units in the LSTM cell. 
The value of it is it up to you, 
too high a value may lead to overfitting or a very low value may yield extremely poor results.
'''

'''
    embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
    inputs = tf.nn.embedding_lookup(embedding, self.input_data)
'''

'''
#TODO:
1. mail Jason and the other guy with doubts in the paper
2. figure out how to train with the list of outputs, we have the labels
3. figure out how to do the embeddings stuff (one-hot, word2vec?)
4. figure out how the embeddings will be trained 
5. figure out how to visualize the net
5a. figure out how an RNN in general works
6. figure out how to use tensorboard (optional)
'''
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from config import Config as conf
import sys # argparse?

def gen_top_words(filename):
    vocabulary = {}
    with open(filename) as f:
        for line in tqdm(f):
            tokens = line.split(" ")
            for token in tokens:
                if token in vocabulary:
                    vocabulary[token] += 1
                else:
                    vocabulary[token] = 1
    temp = [] # extract top words
    for word in vocabulary:
        temp.append([vocabulary[word] , word])
    temp.sort(reverse = True)
    temp = temp[:conf.top_words]
    return [word[1] for word in temp]

def model():    
    input = tf.placeholder(tf.float32, [None, 
        conf.seq_length, conf.embed_size], name = "input")
    
    
    U = tf.get_variable("U", shape=[conf.embed_size, conf.num_hidden_state], 
        initializer=tf.contrib.layers.xavier_initializer())
    V = 
    B = tf.get_variable("B", shape=[conf.vocab_size], 
        initializer=tf.contrib.layers.xavier_initializer())
    
    
    
    cell = tf.contrib.rnn.LSTMCell(conf.num_hidden_state)
    state = cell.zero_state(conf.batch_size, tf.float32)

    Y_preds = []

    with tf.variable_scope("myrnn") as scope: # http://stackoverflow.com/questions/36941382/tensorflow-shared-variables-error-with-simple-lstm-network
        for i in range(conf.seq_length):
            if i > 0:
                scope.reuse_variables() 
            interim, state = cell(input[:,i,:], state) #TODO, inputs? need to feed the ground truth, not our result
            Y_logits = tf.matmul(interim, ) + B
            Y_pred = tf.nn.softmax(Y_logits)
            Y_preds.append(Y_pred)

def main():
    vocabulary = gen_top_words(filename)
    


if __name__ == '__main__':
    # TODO: parsing
    if not tf.__version__.startswith("1.0"):
        print "WARN: Tensorflow version less than 1.0, errors might arise due to API issues"        
    main()