#!/usr/bin/env python
# _*_ coding:utf-8 _*_
"""
数据加载，读取部分
"""
import gc
import json
import os
import re

import cv2 as cv
import numpy as np


from progressbar import ProgressBar

class AbstractData:
    def __init__(self, height, width):
        self.height, self.width = height, width
        self.images = None
        self.labels = None
        self.indices = None


        # label_map:是一个字典，key：字，value:值
        self.label_map = {}
        # key:值，value:字
        self.label_map_reverse = {}
        self.alias_map = {}
        self.alias_map_reverse = {}

        # 随机指数
        self.indices = None
        # ?
        self.batch_ptr = 0

    # 加载char_map:eg：{"的": 0, "日": 1, "军": 2, "者": 3, "意": 4, 。。
    # 字对应的序号，字典
    def load_char_map(self, file_path):
        # end='',表示末尾不换行，因为print默认是打印一行
        print('loading char map from `%s` ...\t' % file_path, end='')
        with open(file_path, encoding='utf-8') as f:
            self.label_map = json.load(f)
        for k, v in self.label_map.items():
            self.label_map_reverse[v] = k
        print('[done]')
        return self

    # 写数据到json中
    def dump_char_map(self, file_path):
        print('Generating char map to `%s` ...\t' % file_path, end='')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.label_map, f, ensure_ascii=False, indent=2)
        print('[done]')
        return self

    # 清空char_map
    def clear_char_map(self):
        self.label_map = {}
        self.label_map_reverse = {}

    # 标点符号的map映射
    def load_alias_map(self, file_path):
        print('loding alise map from `%s` ....\t' % file_path, end='')
        with open(file_path, encoding='utf-8') as f:
            self.alias_map = json.load(f)
        for k, v in self.alias_map.items():
            self.label_map_reverse[v] = k
        print('[done]')
        return self

    def filename2label(selfs, filename: str):
        raise Exception('filename2label not implement')

    # 从文件夹中读取数据
    def read(self, src_root, size=-1, make_char_map=False):
        print('loading data... %s from %s ' % ('' if size == -1 else ("[%d]" % size), src_root))
        # 存放图片
        images = []
        # 标记：列表，存放数字
        labels = []
        with ProgressBar(max_value=None if size == -1 else size) as bar:
            """
            os.walk返回一个生成器，每次遍历返回的对象是一个元组，元组中包含三个元素：
            dirpath:当前遍历的文件夹的路径，类型为字符串；
            dirname:当前遍历的文件夹下的子文件夹的名字，类型为列表；
            filenames:当前遍历的文件夹下的文件的名字，类型为列表；
            """
            for parent_dir, _, filenames in os.walk(src_root, followlinks=True):
                for filename in filenames:
                    # lbl提取出来的就是字
                    lbl = self.filename2label(filename)
                    if make_char_map and lbl not in self.label_map:
                        # next_idx: label_map字典的长度
                        next_idx = len(self.label_map)
                        self.label_map[lbl] = next_idx
                        self.label_map_reverse[next_idx] = lbl
                    labels.append(self.label_map[lbl])
                    images.append(
                        # cv2.imdecode()函数从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式;主要用于从网络传输数据中恢复出图像。
                        cv.imdecode(np.fromfile(os.path.join(parent_dir, filename)), 0)
                        .astype(np.float32)
                        .reshape((self.height, self.width, 1)) / 255.
                    )
                    bar.update(bar.value + 1)
        print('transforming to numpy array...', end=' ')
        self.images = np.array(images)
        self.labels = np.array(labels)
        print('[done]')
        del images
        del labels
        # gc模块，垃圾回收机制
        gc.collect()
        return self

    # 获取图片的个数.images.shape[0]，获取图片的行数
    def size(self):
        return self.images.shape[0]

    # 随机指数
    def shuffle_indices(self):
        print('shuffling...', end=' ')
        samples = self.size()
        # np.random.permutation(samples):随机生成序列
        self.indices = np.random.permutation(samples)
        self.batch_ptr = 0
        print('[done]')
        return self

    # 初始化随机数
    def init_indices(self):
        samples = self.size()
        # 顺序生成0-samples之间的值， indices是一个列表
        self.indices = np.arange(0, samples, dtype=np.int32)
        # 记录开始，结束的位置
        self.batch_ptr = 0
        return self

    # 下一个batch?
    def next_batch(self, batch_size):
        start, end = self.batch_ptr, self.batch_ptr + batch_size
        # 如果结束的地方小于整个indices的长度，那么把新的indices赋值给end
        end = end if end <= len(self.indices) else len(self.indices)
        if start >= self.size():
            return None
        else:
            # 在start和end之间找indices区间的随机数
            """
            indices = [indices[i] for i in range(0, 5)]
            print(indices)
            [12, 15, 1, 7, 4]
            """
            indices = [self.indices[i] for i in range(start, end)]
            self.batch_ptr = end
            return self.images[indices], self.labels[indices]

    # map的逆操作
    def unmap(self, src):
        # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
        # 如果src是str类型，那么返回它的value值，就是字
        if isinstance(src, str):
            return self.label_map_reverse[src]
        else:
            rs = []
            for el in src:
                alias = self.label_map_reverse[el]
                char = alias if alias not in self.alias_map else self.alias_map[alias]
                rs.append(char)
            return rs

    def get(self):
        return self.images, self.labels

class SingleCharData(AbstractData):
    # \d 匹配数字；\w匹配字母或者数字
    ptn = re.compile("\d+_(\w+)\.(?:jpg|png|jpeg)")
    def filename2label(self, filename: str):
        m = SingleCharData.ptn.search(filename)
        return m.group(1) if m else ' '
















