
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq


class Seq2SeqModel(object):

    def __init__(self):
        batch_size = 512
        nodes = 256
        embed_size = 20
        x_seq_length = 26
        self.y_seq_length = 25
        nxchars = 30

        self.ltokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '=', '#leq', '#neq', '#geq', '#alpha',
                            '#beta', '#lambda', '#lt', '#gt', 'x', 'y', '^', '#frac', '{', '}' ,' ']

        # Tensor where we will feed the data into graph
        inputs = tf.placeholder(tf.float32, (None, x_seq_length, nxchars), 'inputs')
        outputs = tf.placeholder(tf.int32, (None, None), 'output')
        targets = tf.placeholder(tf.int32, (None, None), 'targets')

        # Embedding layers
        output_embedding = tf.Variable(tf.random_uniform((len(self.ltokens)+1, embed_size), -1.0, 1.0), name='dec_embedding')
        date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

        with tf.variable_scope("encoding") as encoding_scope:
            lstm_enc = tf.contrib.rnn.BasicLSTMCell(nodes)
            _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=inputs, dtype=tf.float32)

        with tf.variable_scope("decoding") as decoding_scope:
            lstm_dec = tf.contrib.rnn.BasicLSTMCell(nodes)
            dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed, initial_state=last_state)
        #connect outputs to 
        logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=len(self.ltokens)+1, activation_fn=None) 
        with tf.name_scope("optimization"):
            # Loss function
            loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, self.y_seq_length]))
            # Optimizer
            optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)
        self.inputs = inputs
        self.outputs = outputs
        self.logits = logits
    
    def restore(self,sess):
        saver = tf.train.Saver(None)
        path = "seq_mod/model"
        saver.restore(sess, save_path=path)
    
    def batch_data(self, x, y, batch_size):
        shuffle = np.random.permutation(len(x))
        start = 0
        x = x[shuffle]
        y = y[shuffle]
        while start + batch_size <= len(x):
            yield x[start:start+batch_size], y[start:start+batch_size]
            start += batch_size

    def predict_single(self, sess, x):
        x = np.array([x])
        dec_input = np.zeros((len(x), 1)) + len(self.ltokens)
        for i in range(self.y_seq_length):
            batch_logits = sess.run(self.logits,
                        feed_dict = {self.inputs: x,
                        self.outputs: dec_input})
            prediction = batch_logits[:,-1].argmax(axis=-1)
            dec_input = np.hstack([dec_input, prediction[:,None]])
        
        seq = ""
        for c in dec_input[0,1:]:
            c = int(c)
            if c < len(self.ltokens):
                seq += self.ltokens[c] 
                
        return seq

    def get_sequence_data(self, formula, nlabels, bb):
        height, width = formula.shape
        last_xmax = 0
        last_ymin = bb[0]['ymin']
        step_c = -1
        nclasses = nlabels+4+2+1 # 1 for pad and 4 for relative pos, 2 for abs pos and shift from last and width
        seq = np.zeros((30,nclasses))
        for step in bb:
            step_c += 1
            seq[step_c][:nlabels] = step['probs']
            seq[step_c][-1] = 0 # remove pad
            seq[step_c][-7] = step['xmin']/width
            seq[step_c][-6] = step['ymin']/height
            seq[step_c][-5] = (step['xmin']-last_xmax)/10
            last_xmax = step['xmax']
            seq[step_c][-4] = (step['xmax']-step['xmin'])/48
            seq[step_c][-3] = (step['ymin']-last_ymin)/10
            seq[step_c][-2] = (step['ymax']-step['ymin'])/48
            last_ymin = step['ymin']
        return seq