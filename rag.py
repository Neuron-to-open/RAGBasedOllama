# -*- coding: utf-8 -*-
# -------------------------------

# @软件：PyCharm
# @PyCharm：3.4.1
# @Python：3.11
# @项目：COMPUTER SCIENCE IN AI

# -------------------------------

# @文件：rag.py
# @时间：2025/3/11 22:29
# @作者：Neuron-to-opens

# -------------------------------
"""
    1.llm : 用户提问，他回答
        基于：提问，训练数据
    2.RAG : 用户提问，知识库检索相关内容，它回答
        基于: 提问，相关内容，训练数据

    方便检索：转换为向量形式
"""
# 1.创建一个知识库 --- 相似搜索
# 2.实现RAG
import numpy as np

from ollama import chat, Message, embeddings

class Kb :
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # self.content = content
        self.docs = self.split_content(content)
        self.embeds = self.encode(self.docs)

    @staticmethod
    def split_content(content, max_length=50):
        chunks = []
        for i in range(0, len(content), max_length):
            chunks.append(content[i:i+max_length])
        return chunks

    @staticmethod
    def similarity(A, B):
        # 计算点积
        dot_product = np.dot(A, B)
        # 计算范数
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        # 计算余弦相似度
        cosine_similarity = dot_product / (norm_A * norm_B)
        return cosine_similarity

    @staticmethod
    def encode(texts):
        embeds = []
        for text in texts:
            response = embeddings(model='nomic-embed-text', prompt=text)
            embeds.append(response['embedding'])
        return np.array(embeds)

    def search(self, text):
        max_similarity = 0
        max_similarity_index = 0
        e = self.encode([text])[0]
        for idx, te in enumerate(self.embeds) :
            similarity = self.similarity(e, te)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = idx
        return self.docs[max_similarity_index]

class Rag:
    def  __init__(self, model, kb:Kb):
        self.model = model
        self.kb = kb
        self.prompt_template = """
        基于: %s
        回答: %s
        """
    def chat(self, text):
        # 在知识库中查找
        context = self.kb.search(text)
        # 将context拼接到 prompt 里面
        prompt = self.prompt_template % (context, text)
        response = chat(self.model, [Message(role='system', content=prompt)])
        return response['message']


if __name__ == '__main__':
    # kb = Kb('D:\\COMPUTER SCIENCE IN AI\\RAG\\data\\爱因斯坦.txt')
    # # for doc in kb.docs:
    # #     print("=" * 20)
    # #     print(doc)
    # # for e in kb.embed:
    # #     print(e)
    # r = kb.search('相对论')
    #
    # print(r)

    rag = Rag('qwen2.5:7b', Kb('D:\\COMPUTER SCIENCE IN AI\\RAG\\data\\爱因斯坦.txt'))
    while True:
        q = input('Human: ')
        r = rag.chat(q)
        print('Assistant: ', r['content'])
