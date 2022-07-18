import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np

epoch_3 = range(1, 101)
epoch_4 = range(71, 271)
train_loss_3 = []
train_acc_3 = []
train_loss_4 = []
train_acc_4 = []

with open('train_log.txt', 'r', encoding='utf8') as log:
    for i in epoch_3:
        e = log.readline()
        train_loss_3.append(float(log.readline()[6:14]))
        train_acc_3.append(float(log.readline()[15:20]))
    for i in epoch_4:
        e = log.readline()
        train_loss_4.append(float(log.readline()[6:14]))
        train_acc_4.append(float(log.readline()[15:20]))

plt.subplot(1,2,1)
plt.plot(epoch_3, train_loss_3, label='1e-3', color='red')
plt.plot(epoch_4, train_loss_4, label='1e-4', color='blue')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss on train set')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epoch_3, train_acc_3, label='1e-3', color='red')
plt.plot(epoch_4, train_acc_4, label='1e-4', color='blue')
plt.xlabel('epoch')
plt.ylabel('accuracy(%)')
plt.title('accuracy on train set')
plt.legend()
plt.show()

train_loss_4 = np.array(train_loss_4)
train_acc_4 = np.array(train_acc_4)
print('min loss :{:f}'.format(train_loss_4.min()))
print('max acc :{:f}'.format(train_acc_4.max()))