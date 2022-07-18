#### 代码运行环境：

| python版本 | PyTorch版本 | 操作系统     |
| ---------- | ----------- | ------------ |
| 3.7.6      | 1.4.0       | Ubuntu 18.04 |

#### 代码解析：

getWordlist.py：包含了处理分词结果和生成新词表的代码，在进行训练之前，请把数据集放在工作文件夹的`data`子文件夹内，然后运行脚本`getWordlist.py`。

train.py：进行训练的源代码：可以使用resume作为参数，从已经保存的检查点中恢复训练，详情请看代码。

model.py：模型的类定义代码。

cnpre.py：用于保存自定义的`Dataset`。

dotest.ipynb：进行测试的jupyter notebook文件，在可以使用两个模型参数进行句子生成。



checkpoint文件夹中包含了BiLSTM-CRF和LSTM模型，在dotest.ipynb中会用到。

#### 开始训练：

```powershell
python train.py 
```

或者

```pow
python train.py resume
```

