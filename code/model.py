import torch
import torch.nn as nn
from typing import List
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, word_list_size: int, embedding_dim: int, hidden_dim: int, batch_size: int, pad_index: int):
        super().__init__() # 调用nn.Moudle父类的初始化方法
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 优先使用cuda
        
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = word_list_size
        self.pad_index = pad_index

        # 词嵌入层
        self.word_embeds = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = embedding_dim, padding_idx=self.pad_index)

        # 单向LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        # 把初始隐状态和初始细胞状态当作参数，并随机初始化以期望获得良好的效果
        # h0/c0 shape(num_layer*direction, batch, hidden_dim)
        self.h0 = nn.Parameter(torch.randn((1, self.batch_size, self.hidden_dim)))
        self.c0 = nn.Parameter(torch.randn((1, self.batch_size, self.hidden_dim)))

        # 线性层，把LSTM隐藏层输出映射为每个词的概率
        self.hidden2p = nn.Linear(self.hidden_dim, self.vocab_size)

    # 训练集前向传递，返回线性层的概率结果
    def forward(self, sentences: torch.Tensor, sentence_len: int, lens: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeds(sentences)
        packed = pack_padded_sequence(embeds, lens, batch_first=True, enforce_sorted=False)
        # 使用self定义的h0和c0，二者已经在Parameter中，故忽略返回值
        lstm_out, _ = self.lstm(packed, (self.h0, self.c0)) 
        paded, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=sentence_len)
        paded = paded.view(self.batch_size, sentence_len, -1)
        p = self.hidden2p(paded)
        return p
    
    # 获取某个词通过模型之后的输出
    def predict(self, word: int, h, c):
        shape = torch.zeros(self.batch_size, 1)
        word = torch.LongTensor([word]).view(1, 1)
        word = word.expand_as(shape).to(self.device) # 扩充一下word以适应batch
        embeds = self.word_embeds(word)
        lstmout, (new_h, new_c) = self.lstm(embeds, (h, c))
        predict = self.hidden2p(lstmout)
        _, ans = torch.max(predict, 2)
        return ans.view(-1)[0], new_h, new_c

    # 更换计算设备
    def set_device(self, device: torch.device) -> torch.nn.Module:
        _model = self.to(device)
        _model.device = device
        return _model