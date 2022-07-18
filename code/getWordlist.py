from model import *
import torch
from torch import optim
import data
from tqdm import tqdm # 进度条

embedding_dim = 256
lstm_hidden_dim = 128
batch_size = 1

def main():
    # 读入数据训练集
    print('reading data...', end='')
    word2idx, train_set = data.get_data('train')
    tag2idx = data.tag2idx
    _, test_set = data.get_data('test')
    print('\t\tdone.')
    train_size = len(train_set)
    test_size = len(test_set)

    # 建立模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 计算设备
    print('building model...', end='')
    model = bilstm_crf(tag2idx, len(word2idx), embedding_dim, lstm_hidden_dim)
    model.set_device(device)
    print('\tdone.')
    
    # 读取参数
    checkpoint = torch.load('./checkpoint/lr=1e-5.pt')
    model.load_state_dict(checkpoint['net'])

    train_text = []
    with open('data/train.txt', "r", encoding = "utf8") as train:
        for words in train.read().splitlines():
            words = words.replace(" ", "")
            train_text.append(words)
    test_text = open('data/msr_test.utf8', 'r', encoding='utf8').read().splitlines()

    model.eval()
    wordlist = {}
    new_train = []
    bar = tqdm(total = test_size + train_size, leave=False, ascii=True, desc='procssing')
    # 读入测试集
    for j in range(test_size):
        item = test_set[j]
        sentence = item[0]
        sentence= sentence.to(device)
        predict_tag = torch.Tensor(model.test(sentence, batch_size)[0])

        s = test_text[j]
        sp = []
        words = []
        for index in range(len(predict_tag)):
            if predict_tag[index]==data.tag2idx['S'] or predict_tag[index]==data.tag2idx['B']:
               sp.append(index)
        sp.append(len(predict_tag))
        for i in range(len(sp)):
            if i != 0:
                words.append(s[sp[i-1]:sp[i]])
        new_train.append(words)
        for word in words:
            if word in wordlist:
                wordlist[word] += 1
            else:
                wordlist[word] = 1
        bar.update(1)
                  
    # 读入训练集
    for j in range(train_size):
        item = train_set[j]
        sentence = item[0]
        sentence= sentence.to(device)
        predict_tag = torch.Tensor(model.test(sentence, batch_size)[0])

        s = train_text[j]
        sp = []
        words = []
        for index in range(len(predict_tag)):
            if predict_tag[index]==data.tag2idx['S'] or predict_tag[index]==data.tag2idx['B']:
               sp.append(index)
        sp.append(len(predict_tag))
        for i in range(len(sp)):
            if i != 0:
                words.append(s[sp[i-1]:sp[i]])
        new_train.append(words)
        for word in words:
            if word in wordlist:
                wordlist[word] += 1
            else:
                wordlist[word] = 1
        bar.update(1)
    bar.close()
    wordlist['<eos>'] = 1
    wordlist['<pad>'] = 1
                  
    wordlist = sorted(wordlist)
    words = []
    with open('wordlist.txt', "w", encoding='utf8') as result:
        for word in wordlist:
            result.write(word + " ")
            words.append(word) 
    word2idx = dict(zip(words, range(len(words))))
    idx2word = dict(zip(range(len(words)), words))
    torch.save({
        "wordlist": wordlist, # 词表
        "word2idx": word2idx, # 编号之后的词表
        "idx2word": idx2word
    },
        "wordlist.pt")
    torch.save({
        "train_set": new_train # 训练集，包括msr_training和msr_test的分词结果
    },
        'train_set_in_text.pt')
    trainset = torch.load("train_set_in_tag.pt")
    print(trainset["train_set"][0])
    

if __name__ == '__main__':
    main()