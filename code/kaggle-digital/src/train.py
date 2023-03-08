import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

class MyDataset(Dataset):
    def __init__(self,train_data,transform):
        self.train_data = train_df
        self.transform = transform

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        label = self.train_data[idx, 0]  # 第一行为数据类别(标签)
        image = torch.FloatTensor(self.train_data[idx, 1:]).view(28, 28).unsqueeze(0)  # 类型转换, 将list ,numpy转化为tensor
        image = self.transform(image)
        return image, label

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

train_dataset = MyDataset(train_data=train_df,transform=transform)
train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv1 = nn.Sequential(  # (1,32,32)
            nn.Conv2d(
                in_channels=1,  # 输入通道数，若图片为RGB则为3通道
                out_channels=6,  # 输出通道数，即多少个卷积核一起卷积
                kernel_size=5,  # 卷积核大小
                stride=1,  # 卷积核移动步长
                padding=0,  # 边缘增加的像素，使得得到的图片长宽没有变化
            ),  # (6,28,28)
            nn.MaxPool2d(2,stride=1,padding=0) #(6,14,14)
        )
        self.conv2 = nn.Sequential(  # (6,14,14)
            nn.Conv2d(
                in_channels=6,  # 输入通道数
                out_channels=16,  # 输出通道数，即多少个卷积核一起卷积
                kernel_size=5,  # 卷积核大小
                stride=1,  # 卷积核移动步长
                padding=0,  # 边缘增加的像素，使得得到的图片长宽没有变化
            ),  # (16,10,10)
            nn.MaxPool2d(2,stride=1,padding=0) #(16,5,5)
        )
        self.linear = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.Linear(120,84),
            nn.Linear(84,10)
        )

    def forward(self,input):
        input = self.conv1(input)
        input = self.conv2(input)
        output = self.linear(input)
        return output

lenet5 = Lenet5()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lenet5.parameters(),lr=LR)

total_train_step = 0
total_test_step = 0

for epoch in range(EPOCH):
    print("---------------第 {} 轮训练开始---------------".format(epoch+1))
    for data in train_dataloader:
        imgs,targets = data
        outputs = lenet5(imgs)
        loss = loss_fn(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        print("训练次数：{},Loss = {}".format(total_train_step,loss.item()))

torch.save(lenet5.state_dict(),'model_weights.pth')

