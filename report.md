## Natural Language Processing -- LSTM Language Model

<p align = right>18340184 谢善睿</p>

#### Research contents

​	This experiment requires training an LSTM language model based on the word segmentation result from, Chinese word segmentation, the  last experiment. We are required to learn how to use LSTM language model to predict the next word of a Chinese word and generate a sentence.

#### Research plan

1.  Build an LSTM neural network using `pytorch`.
2.  Use the BiLSTM-CRF model of the previous task to partition sentences in `msr_training.utf8` and `msr_test.utf8` and generate a word list. Beside, we use the sentence segmentation result to build a new train set.
3.  Let neural network study on each sentence in trainset to learn the front-back relationship between words.
4.  Use the studied neural network to use the given word to generate a sentene. 

#### Algorithm theory and code explanation

​	In this mission, we use general LSTM neural network. LSTM has three layer: embedding layer, lstm layer and linear layer.

##### embedding layer

​	The word embedding layer works like a fully connected neural network. Its input is the index of a word in the word list, and the output is the word embedding of the word. The process of generating word embedding is using a multi-dimensional vector to uniquely represent the characteristics of a word. In our model, the process of learning word embedding is a part of neural network learning. We use `torch.nn.embedding` class in `PyTorch` to do this work. Embedding layer maps input word, represent in index in word list,  into a low dimensional and dense space.

##### LSTM layer

​	LSTM layer change the word embeddings into hidden states. In LSTM, historical information is called cell state, which is represented by vectors. There are two pathways for LSTM to update cell state, corresponding to two gates:

1.  Forgetting gate: The input of the characteristic vector $x_t$ at the current time and the hidden state $h_{t-1}$ of the previous time are transformed linearly according to the weights $w_{if}$ and $w_{hf}$ and the offsets $b_{if}$ and $b_{hf}$. Then, the vector $f_t$ is obtained through the activation function. The calculation formula is as follows:

$$
f_t=\sigma(W_{if}x_t+b_{if}+W_{hf}h_{t-1}+b_{hf})
$$

2.  Input gate: We generate a vector $i_t$ by making linear transform on the current hidden state vector $h_t$. In addition, we also need to determine the retention degree of $ i_t$ in the historical imformation representing by vector $g_t$, in the input gate. The calculation formula is as follows:
    $$
    i_t=\sigma(W_{ii}x_t + b_{ii} + W_{hi}h_{t-1} + b_{hi})
    $$

    $$
    g_t = tanh(W_{ig}x_t + b_{ig} + W_{hg}h_{t-1} + b_{hg})
    $$

    Based on the vectors $f_t$, $i_t$, $g_t$, we calculate the new cell state as following formual:
    $$
    c_t=f_t\odot c_{t_1}+g_t\odot i_t
    $$
    We can compute the hidden state $ h_t$ by:

$$
o_t=\sigma(w_{io}x_t + b_{io} + w_{ho} + b_{ho})\\h_t = o_t\odot tanh(c_t)
$$

​	The network structure of LSTM is as follows:

<img src="E:/自然语言处理/lab1/report/LSTM.png" alt="LSTM" style="zoom: 33%;" />

##### Linear layer

​	When a word go through  the last two layer, it will be changed into a hidden state vector, and linear layer will transform it into another from we want. In this mission, we are required to deal with a classification problem, so hidden state will be change into a probability vector, which dimension is determinded by the length of word list.

##### Analysis of model code

​	We build LSTM model by the following model:

```python
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
```

The Hyper parameters of our model is 

1.  `word_list_size` determines the dimension of output vector for linear layer 
2.  `embedding_dim` determines the dimension of word embedding.
3.  `hidden_dim` determines the out put feature of LSTM layer.
4.  `batch_size` determines the batch size in batch learing.
5.  `pad_index` determin the pad word in word list. In embedding layer pad word will be mapped into a zero vector.

​	The code of forward pass show as follows:

```python
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
```

In order to get a faster learning speed, we use batch learning to make acceleration. However, each sentence has different length, so we need to use string `"<pad>"` to represent pad word. Before we make sequence go through LSTM layer, we should pack the sequence to avoid pad word impact on result. Here, we use `torch.nn.utils.rnn.pack_padded_sequence` and `torch.nn.utils.rnn.pad_packed_sequence` for LSTM layer.

​	When we need to use our model to generate the next word, we use the code as follows:

```python
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
```

Because we use a word to predict another word, we need to return the hidden state and cell state of the word to the calling code, also we need to transfer the hidden state and cell state of the previous word to LSTM model as parameters.

#### Experiment process

​	In this mission we use cross-entropy-loss as loss function, in `pytorch` framework, we use `torch.nn.CrossEntropyLoss` to calculate it. We use `torch.optim.Adam` as optimizer.

​	We do a test on train set after each epoch of train, and save the parameters of neural net work with the highest accuracy. We start learning with `learning rate = 1e-3` for 100 epoch. After that we found that the loss did not fall steadily, so we change learning rate as 1e-4 and keep training until loss become stable.

##### Result

​	After training, the lowest loss on train set is 0.843282 and we got the highest accuracy is 84.36%. The graph on training show as follows:

![1](1.png)

##### Use model to predict

​	We build instances of BiLSTM-CRF and LSTM model. We use BiLSTM-CRF to segement inut sentence, and use the last `h0`, `c0`of LSTM model and the last word to predict a new word. Repeat the process until new word is `"<eos>"`, which means it is the end of a sentence. Besides, if the last word does not appear in word list, we will use a word instead randomly.

​	Our language model generate sentences as follows:

<img src="2.png" alt="2" style="zoom:60%;" />

#### Bonus

1.  Finish report in English.