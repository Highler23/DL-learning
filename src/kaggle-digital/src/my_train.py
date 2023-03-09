import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torch.autograd import Variable

import pandas as pd
import numpy as np

# 参数设置
EOPCH = 1        # 训练轮数
BATCH_SIZE = 64  # 每批次数据数量
LR = 0.01        # 学习速率

# 模型构建
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # (1,28,28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),      # --> (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # -->(16,14,14)
        )
        self.conv2 = nn.Sequential(     # --> (16,14,14)
            nn.Conv2d(16,32,5,1,2), # 这里用了两个过滤器，将将16层变成了32层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # --> (32,7,7)
        )
        self.out = nn.Linear(32*7*7,10) # 将三维的数据展为2维的数据

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)       # (batch,32,7,7)
        x = x.view(x.size(0),-1)    # (batch,32,7,7)
        # output = F.softmax(self.out(x))
        output = self.out(x)
        return output

cnn = CNN()  # 实例化网络模型
optimzer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

# 加载数据集
train = pd.read_csv('../data/train.csv')
train_labels = torch.from_numpy(np.array(train.label[:]))  # 将数组转换成张量，且二者共享内存，对张量进行修改，原始数组也会相应发生改变
# print(train_labels)  # tensor([1, 0, 1,  ..., 7, 6, 9])
train_data = torch.Tensor(np.array(train.iloc[:,1:]).reshape((-1,1,28,28)))/255  # iloc[row,col] reshape((r,c,w,h))
train_data = TensorDataset(train_data,train_labels)  # TensorDataset(x_train,y_train)  传入数据为tensor

train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print('----------数据加载完成----------')

for epoch in range(EOPCH):
    # print("-" * 25)  # 分界线
    for step,(x,y) in enumerate(train_loader):
        # tensor不能反向传播，variable可以反向传播;variable存放会变化值的地理位置，里面的值会不停变化
        b_x = Variable(x)  # 将tensor转换成variable
        b_y = Variable(y)
        output = cnn(b_x)  # 输出
        loss = loss_func(output,b_y)
        # update W
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        print('epoch %d'%(epoch+1),'start %d'%step)
    print('----------训练结束----------')

# 保存模型
print('开始保存模型......')
torch.save(cnn,'../result/model_weights.pth')
print('模型已保存至: {}'.format('../result/model_weights.pth'))
print('保存模型成功')
print('即将退出程序......')
