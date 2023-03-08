#---------------------------------------------------------------------------------------
# 数据读取
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import pandas as pd
from sklearn.model_selection import train_test_split

data_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
# print(train_df.head(5))
train_df,valid_df = train_test_split(data_df,stratify=data_df.label,test_size=0.2)  # 数据集与验证集之比为8：2
#---------------------------------------------------------------------------------------
# 准备数据集
class NumDataset(Dataset):
    def __init__(self,data_frame,transform,train=True):
        self.data_frame = data_frame.values
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.train == True:
            label = self.data_frame[idx,0]  # 第一行为数据类别(标签)
            image = torch.FloatTensor(self.data_frame[idx,1:]).view(28,28).unsqueeze(0) # 类型转换, 将list ,numpy转化为tensor
            image = self.transform(image)
            return image,label
        else:
            image = torch.FloatTensor(self.data_frame[idx,:]).view(28,28).unsqueeze(0)  # 升维,参数表示在索引处加一个维度
            image = self.transform(image)
            return image
#---------------------------------------------------------------------------------------
# dataloader加载数据集
EPOCH = 30  # 训练轮数
BATCH_SIZE = 64
LR = 0.002  # 学习率
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),std=(0,5,))
])

train_dataset = NumDataset(data_frame=train_df,transform=transform)
train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
valid_dataset = NumDataset(data_frame=valid_df,transform=transform)
valid_dataloader = DataLoader(valid_dataset,batch_size=BATCH_SIZE)
#---------------------------------------------------------------------------------------
# 构建网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # (1, 28, 28)
            nn.Conv2d(
                in_channels=1, # 输入通道数，若图片为RGB则为3通道
                out_channels=32, # 输出通道数，即多少个卷积核一起卷积
                kernel_size=3, # 卷积核大小
                stride=1, # 卷积核移动步长
                padding=1, # 边缘增加的像素，使得得到的图片长宽没有变化
            ),# (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), # (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 池化 (32, 14, 14)
        )
        self.conv3 = nn.Sequential(# (32, 14, 14)
            nn.Conv2d(32, 64, 3, 1, 1),# (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),# (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),# (64, 7, 7)
        )
        self.out = nn.Sequential(
            nn.Dropout(p = 0.5), # 抑制过拟合
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) # (batch_size, 64*7*7)
        output = self.out(x)
        return output
#---------------------------------------------------------------------------------------
# 开始训练
cnn = CNN().to(device)
print(cnn)
# 训练
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_dataloader):
        b_x = x.to(device)
        b_y = y.to(device)
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('epoch:[{}/{}], loss:{:.4f}'.format(epoch, EPOCH, loss))
    with torch.no_grad():
        total = 0
        cor = 0
        for x, y in valid_dataloader:
            x = x.to(device)
            y = y.to(device)
            out = cnn(x)
            pred = torch.max(out, 1)[1]
            total += len(y)
            cor += (y == pred).sum().item()
    print('acc:{:.4f}'.format(cor/total))

