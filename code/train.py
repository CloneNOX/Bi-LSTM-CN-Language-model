import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from cnpre import CNPRE, miniCNPRE
from model import LSTM
import sys

# 超参数
embedding_dim = 300
lstm_hidden_dim = 300
lr = 1e-3
batch_size = 32

# 词编号表
word2idx = torch.load('wordlist.pt')['word2idx']
# 路径
log_path = 'cn-pretend log.txt'
checkpoint_load_path = './checkpoint/cn-pre-check-point lr=1e-4.pt'
checkpoint_save_path = './checkpoint/cn-pre-check-point lr=1e-4.pt'

def main():
    # 读入数据
    print('reading data...', end='')
    sys.stdout.flush()
    train_data = CNPRE() # 数据集
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=batch_padding, drop_last=True) # dataloader
    test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=batch_padding, drop_last=True)
    print("\t\tdone.")

    print('building model...', end='')
    sys.stdout.flush()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') # 计算设备
    pad_index = word2idx['<pad>']
    lstm = LSTM(len(word2idx), embedding_dim, lstm_hidden_dim, batch_size, pad_index)
    lstm.set_device(device)
    print('\tdone.')

    best_accuracy = 0
    start_epoch = 1
    end_epoch = 0
    
    # 处理继续训练的情况
    if len(sys.argv) == 2 and sys.argv[1] == 'resume':
        log = open(log_path, 'a')
        log.write('====================================================== lr = 1e-4\n')
        checkpoint = torch.load(checkpoint_load_path) # 上一个检查点保存路径
        lstm.load_state_dict(checkpoint['net'])
        best_accuracy = checkpoint['accuracy']
        start_epoch = checkpoint['epoch'] + 1
        end_epoch = start_epoch + 50
        lr = 1e-4
        log.close()
    elif len(sys.argv) == 1:
        log = open(log_path, 'w')
        log.close()
        lr = 1e-3
        start_epoch = 1
        end_epoch = start_epoch + 50

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=word2idx['<pad>']) # 损失函数（交叉熵）
    optimizer = optim.Adam(lstm.parameters(), lr=lr) # 优化器

    for epoch in range(start_epoch, end_epoch):
        print('epoch {:>3d}'.format(epoch))
        with open(log_path, 'a') as log:
            log.write('epoch {:>3d}\n'.format(epoch))

        # 训练集通过模型，并且优化
        lstm.train()
        train_loss = 0.
        train_pbar = tqdm(total=len(train_loader), leave=False, ascii=True, desc='traning', unit='batch') # 进度条
        for tags, anss, maxlen, lens in train_loader:
            tags = tags.to(device)
            anss = anss.view(-1).to(device)
            lens = lens.to(device)
            optimizer.zero_grad()

            pre = lstm.forward(tags, maxlen, lens).view(batch_size * maxlen, len(word2idx))
            loss = loss_func(pre, anss)
            loss.backward()
            optimizer.step()
            
            train_loss += loss
            train_pbar.update(1) # 更新进度条
        train_pbar.close()
        train_loss = train_loss / (len(train_data) // batch_size)
        print("loss: {:.6f}".format(train_loss))
        with open(log_path, 'a') as log:
            log.write("loss: {:.6f}\n".format(train_loss))
        
        # 在训练集上测试(QAQ)
        lstm.eval()
        total_word = 0
        correct_word = 0
        train_pbar = tqdm(total=len(train_loader), leave=False, ascii=True, desc='testing', unit='batch') # 进度条
        for tags, anss, maxlen, lens in train_loader:
            tags = tags.to(device)
            anss = anss.to(device)
            lens = lens.to(device)

            pre = lstm.forward(tags, maxlen, lens).view(batch_size, maxlen, len(word2idx))
            max_index = torch.max(pre, 2)[1] # Tensor shape(batch_size, maxlen) 最大概率的下标
            correct_word += (anss == max_index).sum().item()
            total_word += lens.sum().item()
            train_pbar.update(1) # 更新进度条
        train_pbar.close()
        print('test accuracy: {:.2f}%, ({:d}/{:d})'.format(100 * correct_word/total_word, correct_word, total_word))

        with open(log_path, 'a') as log:
            log.write('test accuracy: {:.2f}%, ({:d}/{:d})\n'.format(100 * correct_word/total_word, correct_word, total_word))

        accuracy = correct_word/total_word
        # 保存最好的模型
        if accuracy > best_accuracy:
            print('saving best checkpoint...\n')
            state = {
                'net': lstm.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch
            }
            torch.save(state, checkpoint_save_path) # 保存这个检查点
            best_accuracy = accuracy


# pad batch内的数据并Tensor化
def batch_padding(batch):
    maxlen = 0
    for item in batch:
        maxlen = max(maxlen, len(item[0]))
    tags = []
    anss = []
    lens = []
    for item in batch:
        lens.append(len(item[0]))
        for i in range(len(item[0]), maxlen):
            item[0].append(word2idx['<pad>'])
            item[1].append(word2idx['<pad>'])
        tags.append(item[0])
        anss.append(item[1])
    tags = torch.LongTensor(tags)
    anss = torch.LongTensor(anss)
    lens = torch.IntTensor(lens)
    return tags, anss, maxlen, lens

if __name__ == '__main__':
    main()