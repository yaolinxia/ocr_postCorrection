#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf

class Model:
    # 定义模型输入的宽度，高度类的数量，模式（train, test）
    def __init__(self, input_width, input_height, num_class, mode):
        # 初始化
        self.input_width = input_width
        self.input_height = input_height
        self.num_class = num_class

        self.training = mode.lower() in ('train',)

        self.images = tf.placeholder(tf.float32, [None, input_height, input_width, 1], name='input_img_batch')
        self.labels = tf.placeholder(tf.int32, [None], name='input_lbl_batch')

        # 相关操作
        self.step = None
        self.loss = None
        self.classes = None

        # 预测的结果
        self.prob = None
        self.train_op = None
        self.val_acc = None
        self.val_acc_update_op = None

        # add
        self.num_layers = None
        self.num_hidden = None
        self.data = None

    # 存储数据
    def feed(self, images, labels):
        return {
            self.images: images,
            self.labels: labels
        }

    # 模型构建
    def build(self):
        images = self.images
        label = self.labels

        cells = list()

        for _ in range(self.num_layers):
            cell = tf.nn.rnn_cell.GRUCell(num_units=self.num_hidden)
            """
            第一个就是输入的循环神经网络的cell，可以设定为BasicLSTMCell等等。
            第二个参数就是输入数据使用dropout，后面的概率，如果是一，就不会执行dropout。dropout:防止过拟合
            """
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=1.0)
            cells.append(cell)
        network = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
        outputs, last_state = tf.nn.dynamic_rnn(cell=network, inputs=self.data, dtype=tf.float32)

        # get last output
        outputs = tf.transpose(outputs, (1, 0, 2))
        last_output = tf.gather(outputs, int(outputs.get_shape()[0]-1))

        # linear transform
        out_size = int(target.)
