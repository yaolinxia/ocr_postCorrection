#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np

def output_state():
    # tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值。
    X = tf.random_normal(shape=[3, 5, 6], dtype=tf.float32)
    # print(X)
    X = tf.reshape(X, [-1, 5, 6])
    # print(X)                                              `
    """
    tf.nn.rnn_cell.GRUCell(num_units, input_size=None, activation=<function tanh>).num_units
    就是隐层神经元的个数，默认的activation就是tanh，你也可以自己定义，但是一般都不会去修改。
    这个函数的主要的参数就是num_units。
    """
    cell = tf.nn.rnn_cell.GRUCell(10)
    # print(cell)
    init_state = cell.zero_state(3, dtype=tf.float32)
    # print(init_state)
    output, state = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, time_major=False)
    with tf.Session() as sess:
        print(sess.run(X))
        sess.run(tf.initialize_all_variables())
        print("state===========================================")
        print(sess.run(state))
        print("output==============================================")
        print(sess.run(output))
        # print(sess.run(init_state))

def test_shape():
    # images = tf.placeholder(tf.float32, [None, 64, 64, 1], name='input_img_batch')
    images = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    d = images.reshape((-1, 1, 2, 1))
    print(d)           
    # input_layer = tf.reshape(images, [-1, 1, 1, 1])
    # print(input_layer)
    # with tf.Session() as sess:
    #     print(sess.run(input_layer))
        # print("input_layer", input_layer)

def test_gru(size):

    # 定义输入
    input_data = tf.placeholder(tf.float32, [-1, 64*64, 1], name='input_img_batch')
    # 定义每一个cell的结构
    gru_cell = tf.nn.rnn_cell.GRUCell(size)
    # output:输出，state:状态
    # last_states是最终的状态，而outputs对应的则是每个时刻的输出。
    outputs, last_states = tf.nn.dynamic_rnn(cell=gru_cell, dtype=tf.float64, sequence_length=)




if __name__ == '__main__':
    print("+=========================")
    test_shape()