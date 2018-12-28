#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf

class Model:

    # 定义模型输入的宽度，高度，类的数量，模式(train, test)
    def __init__(self, input_width, input_height, num_class, mode):
        # 初始化
        self.input_width = input_width
        self.input_height = input_height
        self.num_class = num_class
        self.training = mode.lower() in ('train',)

        self.images = tf.placeholder(tf.float32, [None, input_height, input_width, 1], name='input_img_batch')
        self.labels = tf.placeholder(tf.int32, [None], name='input_lbl_batch')

        # 定义一些操作
        self.step = None
        self.loss = None
        # self.logits = None
        self.classes = None
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

    # 模型构建
    def build(self):
        images = self.images
        labels = self.labels

        # 对输入的矩阵进行形状的改变，根据图片的宽，高决定
        input_layer = tf.reshape(images, [-1, self.input_height, self.input_width, 1])

        # cnn block 1
        x = tf.layers.conv2d(
            # 输入就是图像，Tensor输入
            inputs=input_layer,
            # filters:整数，表示输出空间的维数（卷积核过滤的数量）
            filters=32,
            # 卷积核的窗和宽
            kernel_size=3,
            # padding='same'：不够卷积核的块补0
            padding='same',
            # 卷积核的初始化
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        # 提高速度
        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            x,
            pool_size=[2, 2],
            strides=2
        )

        # cnn block 2
        x = tf.layers.conv2d(
            # 输入就是图像，Tensor输入,上一块的输出结果
            inputs=x,
            # filters:整数，表示输出空间的维数（卷积核过滤的数量）
            filters=64,
            # 卷积核的窗和宽
            kernel_size=3,
            # padding='same'：不够卷积核的块补0
            padding='same',
            # 卷积核的初始化
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        # 提高速度
        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            x,
            pool_size=[2, 2],
            strides=2
        )

        # cnn block 3
        x = tf.layers.conv2d(
            # 输入就是图像，Tensor输入,上一块的输出结果
            inputs=x,
            # filters:整数，表示输出空间的维数（卷积核过滤的数量）
            filters=128,
            # 卷积核的窗和宽
            kernel_size=3,
            # padding='same'：不够卷积核的块补0
            padding='same',
            # 卷积核的初始化
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        # 提高速度
        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            x,
            pool_size=[2, 2],
            strides=2
        )

        # cnn block 4
        x = tf.layers.conv2d(
            # 输入就是图像，Tensor输入,上一块的输出结果
            inputs=x,
            # filters:整数，表示输出空间的维数（卷积核过滤的数量）
            filters=256,
            # 卷积核的窗和宽
            kernel_size=3,
            # padding='same'：不够卷积核的块补0
            padding='same',
            # 卷积核的初始化
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        # 提高速度
        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            x,
            pool_size=[2, 2],
            strides=2
        )

        # 加一个全连接层
        x = tf.layers.flatten(
            inputs=x
        )
        # 搞了两个全连接层
        x = tf.layers.dense(
            inputs=x,
            units=8192
        )

        # 逻辑层
        self.logits = tf.layers.dense(
            inputs=x,
            # units=self.num_class:该层的神经单元结点数
            units=self.num_class
        )

        self.prob = tf.nn.softmax(self.logits, name='P')
        self.classes = tf.argmax(input=self.prob, axis=1, name='class')
        self.step = tf.train.get_or_create_global_step()

        # 使用交叉熵损失
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits)
        # 使用优化器
        optimizer = tf.train.AdamOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(
                loss=self.loss,
                global_step = self.step
            )

        return self

        # self.val_acc, self.val_acc_up


#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf

class Model:

    # 定义模型输入的宽度，高度，类的数量，模式(train, test)
    def __init__(self, input_width, input_height, num_class, mode):
        # 初始化
        self.input_width = input_width
        self.input_height = input_height
        self.num_class = num_class
        self.training = mode.lower() in ('train',)
        self.images = tf.placeholder(tf.float32, [None, input_height, input_width, 1], name='input_img_batch')
        self.labels = tf.placeholder(tf.int32, [None], name='input_lbl_batch')

     # 存储数据
    def feed(self, images, labels):
        return {
            self.images: images,
            self.labels: labels
        }

    # 模型构建
    def build(self):
        images = self.images
        labels = self.labels

        # 对输入的矩阵进行形状的改变，根据图片的宽，高决定
        input_layer = tf.reshape(images, [-1, self.input_height, self.input_width, 1])

        # cnn block 1
        x = tf.layers.conv2d(
            # 输入就是图像，Tensor输入
            inputs=input_layer,
            # filters:整数，表示输出空间的维数（卷积核过滤的数量）
            filters=32,
            # 卷积核的窗和宽
            kernel_size=3,
            # padding='same'：不够卷积核的块补0
            padding='same',
            # 卷积核的初始化
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        # 提高速度
        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            x,
            pool_size=[2, 2],
            strides=2
        )

        # cnn block 2
        x = tf.layers.conv2d(
            # 输入就是图像，Tensor输入,上一块的输出结果
            inputs=x,
            # filters:整数，表示输出空间的维数（卷积核过滤的数量）
            filters=64,
            # 卷积核的窗和宽
            kernel_size=3,
            # padding='same'：不够卷积核的块补0
            padding='same',
            # 卷积核的初始化
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        # 提高速度
        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            x,
            pool_size=[2, 2],
            strides=2
        )

        # cnn block 3
        x = tf.layers.conv2d(
            # 输入就是图像，Tensor输入,上一块的输出结果
            inputs=x,
            # filters:整数，表示输出空间的维数（卷积核过滤的数量）
            filters=128,
            # 卷积核的窗和宽
            kernel_size=3,
            # padding='same'：不够卷积核的块补0
            padding='same',
            # 卷积核的初始化
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        # 提高速度
        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            x,
            pool_size=[2, 2],
            strides=2
        )

        # cnn block 4
        x = tf.layers.conv2d(
            # 输入就是图像，Tensor输入,上一块的输出结果
            inputs=x,
            # filters:整数，表示输出空间的维数（卷积核过滤的数量）
            filters=256,
            # 卷积核的窗和宽
            kernel_size=3,
            # padding='same'：不够卷积核的块补0
            padding='same',
            # 卷积核的初始化
            kernel_initializer=tf.glorot_uniform_initializer()
        )

        # 提高速度
        x = tf.layers.batch_normalization(
            inputs=x,
            training=self.training
        )

        x = tf.nn.leaky_relu(x, alpha=0.01)

        x = tf.layers.max_pooling2d(
            x,
            pool_size=[2, 2],
            strides=2
        )

        # 加一个全连接层
        x = tf.layers.flatten(
            inputs=x
        )
        # 搞了两个全连接层
        x = tf.layers.dense(
            inputs=x,
            units=8192
        )

        # 逻辑层
        self.logits = tf.layers.dense(
            inputs=x,
            # units=self.num_class:该层的神经单元结点数
            units=self.num_class
        )

        self.prob = tf.nn.softmax(self.logits, name='P')
        self.classes = tf.argmax(input=self.prob, axis=1, name='class')
        self.step = tf.train.get_or_create_global_step()

        # 使用交叉熵损失
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits)
        # 使用优化器
        optimizer = tf.train.AdamOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(
                loss=self.loss,
                global_step = self.step
            )

        self.val_acc, self.val_acc_update_op = tf.metrics.accuracy(labels, self.classes)

        return self

        # self.val_acc, self.val_acc_up


