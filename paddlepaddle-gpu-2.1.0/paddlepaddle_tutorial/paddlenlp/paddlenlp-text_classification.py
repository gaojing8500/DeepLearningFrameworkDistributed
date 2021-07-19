'''
Author: your name
Date: 2021-07-16 10:22:51
LastEditTime: 2021-07-16 14:29:38
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /DeepLearningFramework/paddlepaddle-gpu-2.1.0/paddlepaddle_tutorial/paddlenlp/paddlenlp-text_classification.py
'''

import paddle
from paddle.fluid.layers.nn import embedding
import paddlenlp
from paddlenlp.embeddings import TokenEmbedding
import paddle.nn as nn
from data import Tokenizer

# token_embedding = TokenEmbedding(embedding_name="w2v.wiki.target.word-word.dim300")

# text_encode = paddlenlp.seq2vec.BoWEncoder(token_embedding.embedding_dim)

# test_text = ["你好呀！","无聊最为致命","上海联影是最牛逼的公司"]

# text_embedding_list = []


# for text in test_text:
#     text_embedding_text = token_embedding(text)
#     text2seq = text_encode(text_embedding_text)
#     text_embedding_list.append(text_embedding_text)

# print(text_embedding_list[0])

class BoWModel(nn.Layer):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model
        self.encoder = paddlenlp.seq2vec.BoWEncoder(self.embedding_model.embedding_dim)
        self.cosim_fuc = nn.CosineSimilarity(axis=-1)

    def forward(self,text):
       embedding_text = self.embedding_model(text)
       summed = self.encoder(embedding_text)
       
       return summed
    
    def get_cos_sim(self,text_a,text_b):
        text_a_embedding = self.forward(text_a)
        text_b_embedding = self.forward(text_b)
        cos_sim = self.cosim_fuc(text_a_embedding,text_b_embedding)
        return cos_sim

    def word_to_vec(self,text):
        text_vec = self.forward(text)
        print(text_vec)


    def set_vocab_word(self):
        tokenier = Tokenizer()
        tokenier.set_vocab(self.embedding_model.vocab)
        return tokenier

    def word_to_id(self,text_list):
        tokenizer = self.set_vocab_word()
        for text in text_list:
            text_id = paddle.to_tensor([tokenizer.text_to_ids(text)])
            print(text_id)
            self.word_to_vec(text_id)


if __name__  == "__main__":
    token_embedding = TokenEmbedding(embedding_name="w2v.wiki.target.word-word.dim300")
    model = BoWModel(token_embedding)
    test_text = ["你好呀！","无聊最为致命","上海联影是最牛逼的公司"]
    model.word_to_id(test_text)


        




