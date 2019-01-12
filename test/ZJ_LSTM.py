#!/usr/bin/env python
# _*_ coding:utf-8 _*_

#!/usr/bin/python
# -*- coding:UTF-8 -*-
import tensorflow as tf
from attention import attention

class Model(object):
    def rnn_cell(self,FLAGS,dropout):
        single_cell=tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden_units,forget_bias=1.0,initializer=tf.glorot_normal_initializer())
        single_cell=tf.nn.rnn_cell.DropoutWrapper(single_cell,output_keep_prob=dropout)
        return single_cell

    def __init__(self,FLAGS,embedding_matrix):
        self.inputs_X=tf.placeholder(tf.int32,shape=[None,None],name='inputs_X')
        self.targets_y=tf.placeholder(tf.float32,shape=[None,None],name='targets_y')
        self.seq_lens=tf.placeholder(tf.float32,shape=[None,],name='seq_lens')
        self.dropout=tf.placeholder(tf.float32)
        self.topic_vector = tf.placeholder(tf.float32, shape=[None, None], name='topic_vector')
        self.global_step = tf.Variable(0, trainable=False)
        with tf.device('/cpu:0'),tf.name_scope("embedding"):
            self.embedding=tf.Variable(initial_value=embedding_matrix,dtype=tf.float32,name="Embedding",trainable=True)
            inputs=tf.nn.embedding_lookup(self.embedding,self.inputs_X)

        stacked_cell=tf.nn.rnn_cell.MultiRNNCell([self.rnn_cell(FLAGS,self.dropout) for _ in range(FLAGS.num_layers)])

        initial_state=stacked_cell.zero_state(FLAGS.batch_size,tf.float32)
        all_outputs,state=tf.nn.dynamic_rnn(initial_state=initial_state,cell=stacked_cell,inputs=inputs,sequence_length=self.seq_lens,dtype=tf.float32)

        out_before_attention=tf.summary.histogram("out_before_attention",all_outputs)

        with tf.name_scope('attention_layer'):
            outputs_attention=attention(all_outputs,256,self.topic_vector,time_major=False)

        #outputs_attention=outputs_attention/self.seq_lens[:,None]
        #outputs_attention=tf.reduce_sum(all_outputs,1)/self.seq_lens[:,None]
        out_after_attention = tf.summary.histogram("out_after_attention", outputs_attention)
        logits = tf.layers.dense(inputs=outputs_attention, units=FLAGS.num_classes,act