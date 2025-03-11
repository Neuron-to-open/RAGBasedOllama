# -*- coding: utf-8 -*-
# -------------------------------

# @软件：PyCharm
# @项目：COMPUTER SCIENCE IN AI

# -------------------------------

# @文件：rag_from_scratch.py
# @时间：2025/3/11 22:13
# @作者：Neuron-to-opens

# -------------------------------



from ollama import chat, Message

msgs = [
    Message(role="system", content="将用户的输入翻译为英文" ),
    Message(role="user", content="不要温顺的走进那个良夜!")
]


response = chat(model="qwen2.5:7b", messages=msgs)

print(response)