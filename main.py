# LSTM (RNNの一部)
## 時系列データを解析
## RNNは最初の方のモデルが最後に反映しなくなるので、LSTMは内部にメモリセルという機構をもうけ、RNNの欠点を緩和している

from janome.tokenizer import Tokenizer
from gensim.models.keyedvectors import KeyedVectors
import torch.nn as nn

#         | 形容詞       | 名詞       | 助詞       　　|動詞         | 句読点
#     |--------| C1 |--------| c2 |--------| c3 |--------| c4 |--------| c5
# C0→→|        |→→→→|        |→→→→|        |→→→→|        |→→→→|        |→→→→
#     |   NN   |    |   NN   |    |   NN   |    |   NN   |    |   NN   |
# h0→→|        |→→→→|        |→→→→|        |→→→→|        |→→→→|        |→→→→
#     |--------| h1 |--------| h2 |--------| h3 |--------| h4 |--------| h5
#         |             |              |            |            |
#        大きな          犬             が           走る　　　　　　　・


# LSTM
tkz = Tokenizer()
s = "私は犬が大好き"
ws = [w for w in tkz.tokenize(s, wakati=True)]
w2v = KeyedVectors.load_word2vec_format(
    'entity_vector.model.bin', binary = True
)

import numpy as np
import torch

xn = torch.tensor([w2v[w] for w in ws])
print(xn.shape) # 200次元のベクトル

xn = xn.unsqueeze(0)
print(xn.shape) # 6 * 200 の行列

# LSTMを作成
# 第一引数　入力ベクトルのサイズ、LSTMの出力ベクトルのサイズsaizu batchサイズの値を最初に持ってくるので、True
lstm = nn.LSTM(200, 200, batch_first=True)
h0 = torch.randn(1, 1, 200) #省略可
c0 = torch.randn(1, 1, 200)
yn, (hn, cn) = lstm(xn, (h0, c0))# lstmの出力
print(yn.shape)
print(yn.shape)
print(cn.shape)

# 品詞タグ付け

#                       ↑ yi
#                   |--------|
#                   |        |
#                   |    W   |
#                   |(100次元)|
#                   |--------|
#                       |
#                       |
#            C(i-1) |--------| ci
#               →→→→|        |→→→→
#                   |  LSTM  |
#               →→→→|        |→→→→
#            h(i-1) |--------| hi
#                       |
#                       |
#                   |--------|
#                   |        |
#                   |  Embd  | 　分散表現に変換
#                   |        |
#                   |--------|
#                       ↑  Xi

labels = {'名詞': 0, '助詞': 1, '形容詞':2, '助動詞': 3, '補助記号':4, '動詞':5, '代名詞': 6, '接尾辞': 7, '副詞': 8, '形状詞': 9, '記号': 10, '連体詞': 11,
          '接頭辞': 12, '接続詞': 13, '感動詞': 14, ' 空白': 15}

