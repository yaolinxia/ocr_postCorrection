#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import yaml

config_path = './configs/train.yaml'
f = open(config_path, encoding='utf-8')
args = yaml.load(f.read())
f.close()