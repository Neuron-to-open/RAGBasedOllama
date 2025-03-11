# -*- coding: utf-8 -*-
# -------------------------------

# @软件：PyCharm
# @PyCharm：3.4.1
# @Python：3.11
# @项目：COMPUTER SCIENCE IN AI

# -------------------------------

# @文件：embed.py
# @时间：2025/3/11 22:50
# @作者：Neuron-to-opens

# -------------------------------
from ollama import embeddings

response = embeddings(model='nomic-embed-text', prompt='hello')

# print(response)

print(response['embedding'])

print(len(response['embedding']))

print(type(response['embedding']))