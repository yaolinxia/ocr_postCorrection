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

    # 存储数据
    def feed(self, images, labels):
        return {
            self.images : images,
            self.labels : labels
        }

    def build(self):
