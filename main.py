#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf

from models.cnn_model import Model
from data import SingleCharData as Data
from args import args
import os

class Main:
    def __init__(self):

        # 指派设备，是否使用GPU
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = None

        self.infer_model = None

    def run(self, mode):
        self.sess.run(tf.global_variables_initializer())
        if mode in ('train', ):
            self.train()
        elif mode in ('infer', 'pred'):
            self.infer(dump=True)
        else:
            print('mode `%s` cannot be recognized' % mode)

    # 从最近的checkpoint重启
    def restore(self, ckpt_dir=None):
        # tf.train.Saver:管理模型中的所有变量，可以将变量保存到检查点文件中
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        ckpt_dir = ckpt_dir or args['ckpt']
        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        # 如果ckpt存在，那么重新进行存储
        if ckpt:
            self.saver.restore(self.sess, ckpt)
            print('sucessfully restored from %s' % ckpt_dir)
        else:
            print('cannot restore from %s' % ckpt_dir)

    # 每step间隔，保存一下
    def save(self, step):
        if self.saver is None:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        self.saver.save(self.sess, os.path.join(args['ckpt'], '%s_model' % str(args['name'])), global_step=step)
        print('ckpt saved')

    def train(self):
        # 调用cnn模型
        model = Model(args['input_width'], args['input_height'], args['num_class'], 'train')
        model.build()
        # 所有变量进行一个初始化
        self.sess.run(tf.global_variables_initializer())
        # val_data: 加载验证集的数据
        val_data = Data(args['input_height'], args['input_width'])
        # 如果charmap存在, 则直接加载charmap
        if args['charmap_exist']:
            val_data.load_char_map(args['charmap_path'])
        # dir_val: 验证集的数据
        val_data.read(args['dir_val'], size=args['train_size'], make_char_map=not args['charmap_exist'])
        # 如果charmap不存在，生成charmap
        if not args['charmap_exist']:
            val_data.dump_char_map(args['charmap_path'])

        # 加载训练集的数据
        train_data = Data(args['input_height'], args['input_width']) \
            .load_char_map(args['charmap_path']) \
            .read(args['dir_train'], size=args['train_size'], make_char_map=True) \
            .shuffle_indices()
        print('start training')

        # 如果需要从最近的checkpoint重启
        if args['restore']:
            self.restore()

        # 初始化tensorboard
        writer = tf.summary.FileWriter(args['tb_dir'])

        # start training
        step = 0
        cost_between_val = 0
        samples_between_val = 0
        batch_size = args['batch_size']
        for itr in range(args['num_epochs']):
            train_data.shuffle_indices()
            train_batch = train_data.next_batch(batch_size)
            while train_batch is not None:
                images, labels = train_batch
                feed_dict = model.feed(images, labels)
                step, loss, _ = self.sess.run([model.step, model.loss, model.train_op],
                                              feed_dict=feed_dict)
                train_batch = train_data.next_batch(batch_size)
                cost_between_val += loss
                samples_between_val += batch_size

                # 取余为1的时候，保存
                if step % args['save_interval'] == 1:
                    self.save(step)
                # 取余为0的时候，继续
                if step % args['val_interval'] == 0:
                    print("#%d[%d]\t\t" % (step, itr), end='')
                    # val_data:初始化随机数
                    val_data.init_indices()
                    # 选择val_batch
                    val_batch = val_data.next_batch(batch_size)

                    self.sess.run(tf.local_variables_initializer())
                    acc = 0.0
                    val_cost = val_samples = 0
                    while val_batch is not None:
                        # val_batch存放图片，及相应的标签
                        val_images, val_labels = val_batch
                        val_feed_dict = model.feed(val_images, val_labels)
                        loss, _acc, acc = self.sess.run(
                            [model.loss, model.val_acc_update_op, model.val_acc],
                            feed_dict=val_feed_dict
                        )
                        val_cost += loss
                        val_samples += batch_size
                        val_batch = val_data.next_batch(batch_size)
                    loss = val_cost / val_samples
                    # tensorboard部分; acc准确率
                    custom_sm = tf.Summary(
                        value=[tf.Summary.Value(tag='accuracy', simple_value=acc)]
                    )
                    writer.add_summary(custom_sm, step)
                    print("#validation:accuracy=%.6f,\t average_batch_loss:%.4f" % (acc, loss))
                    cost_between_val = samples_between_val = 0
        self.save(step)


def main(_):
    print('using tensorflow', tf.__version__)
    m = Main()
    if args['gpu'] == -1:
        dev = '/cpu:0'
    else:
        dev = '/gpu:%d' % args['gpu']

    with tf.device(dev):
        m.run(args['mode'])

if __name__ == '__main__':
    tf.logging.set_verbosity('ERROR')
    tf.app.run()






