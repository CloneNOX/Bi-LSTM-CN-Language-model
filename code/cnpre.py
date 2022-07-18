import torch
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader

# 获取训练集，返回训练样例的tag以及ans的List组成的元组的List
def get_train_set():
    train_set = torch.load('train_set_in_text.pt')['train_set']
    word2idx = torch.load('wordlist.pt')['word2idx']
    train_data = []
    for sentence in train_set:
        tag = []
        ans = []
        for i in range(len(sentence)):
            tag.append(word2idx[sentence[i]] if sentence[i] in word2idx else word2idx['<pad>']) # 防止出错
            if i < (len(sentence) - 1):
                ans.append(word2idx[sentence[i+1]] if sentence[i+1] in word2idx else word2idx['<pad>'])
            else:
                ans.append(word2idx['<eos>']) # 最后一个词语的ans是'<eos>'
        train_data.append((tag, ans))
    return train_data

def get_mini_set():
    train_set = torch.load('train_set_in_text.pt')['train_set']
    word2idx = torch.load('wordlist.pt')['word2idx']
    train_data = []
    for j in range(4096):
        sentence = train_set[j]
        tag = []
        ans = []
        for i in range(len(sentence)):
            tag.append(word2idx[sentence[i]] if sentence[i] in word2idx else word2idx['<pad>']) # 防止出错
            if i < (len(sentence) - 1):
                ans.append(word2idx[sentence[i+1]] if sentence[i+1] in word2idx else word2idx['<pad>'])
            else:
                ans.append(word2idx['<eos>']) # 最后一个词语的ans是'<eos>'
        train_data.append((tag, ans))
    return train_data

class CNPRE(Dataset):
    def __init__(self):
        # 获得训练集的词标号和答案 list[tuple(tensor, tensor)]
        self.train_set = get_train_set()
    def __getitem__(self, index):
        tag = self.train_set[index][0]
        ans = self.train_set[index][1]
        return tag, ans
    def __len__(self):
        return len(self.train_set)
    
class miniCNPRE(Dataset):
    def __init__(self):
        # 获得训练集的词标号和答案 list[tuple(tensor, tensor)]
        self.train_set = get_mini_set()
    def __getitem__(self, index):
        tag = self.train_set[index][0]
        ans = self.train_set[index][1]
        return tag, ans
    def __len__(self):
        return len(self.train_set)