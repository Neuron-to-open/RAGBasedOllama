# -*- coding: utf-8 -*-
# -------------------------------

# @软件：PyCharm
# @PyCharm：3.4.1
# @Python：3.11
# @项目：RAG

# -------------------------------

# @文件：test.py
# @时间：2025/3/12 16:52
# @作者：Neuron-to-opens

# -------------------------------
import yaml

with open('config/config.yaml', 'r', encoding='utf-8') as f :
    config = yaml.safe_load(f)


print(config['config'])