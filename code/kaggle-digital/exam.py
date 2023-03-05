#-------------------------------------------------------------------------------------------------------
# 数据预处理
import pandas as pd

train_data = pd.read_csv("./data/train.csv")  # 读取训练数据集数据 相对路径
# print(train_data.head(10))  # 查看前十行数据

# dfresult= pd.DataFrame()  # 赋值一个空的表格，dataframe，用于保存数据
label_df = train_data["label"]
feature_df = train_data.drop("label",axis=1)  # 删除训练数据集中的label列

feature_df = feature_df/255.0  # 归一化处理
feature_df = feature_df.apply(lambda x:x.values.reshape(1,28,28), axis=1)  # 进行数据变换，变换成1*28*28(C*H*W)的图像输入形式
# print(feature_df.head(1))
#-------------------------------------------------------------------------------------------------------
# 数据集
import torch
from torch.nn import Module
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

class DigitRecongnizerDataset(Dataset):
    def __init__(self, label_df, feature_df, transform=None, target_transforms=None):
        self.label_df = label_df
        self.images = feature_df
        self.transfrom = transform
        self.target_transforms = target_transforms

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.label_df[idx]
        # 变换 如果传入的话
        if self.transfrom:
            self.transfrom(image)
        if self.target_transforms:
            self.target_transforms(label_df)
        return label,image

    def __len__(self):
        return len(label_df)

drDataset = DigitRecongnizerDataset(label_df, feature_df, transform=transforms.ToTensor())  # 实例化

# 可视化展示一下图片
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(8, 8))  # 一英寸2.5cm，那不就是20cm，怎么容纳的下图片的 A：一厘米容纳25像素
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(drDataset), size=(1,)).item()  # 随机选取数据
    label, img = drDataset[sample_idx]  # 返回取出的数据及其标签
    figure.add_subplot(rows, cols, i)  # subplot布局  (3,3,i)
    plt.title(label)
    plt.axis("off") # 不显示坐标轴
    # squeeze是降维，去除度数1的维度，如灰度图像中C(通道)=1，绘制图像时不需要通道C这个维度，直接传递二维矩阵即可，所以将其去除
    # 但这里由于img是28*28的矩阵不带C这个维度，所以不需要squeeze(),所以不需要squeeze在这里不起作用
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
#-------------------------------------------------------------------------------------------------------
# 模型构建
from torch import nn

class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积池化层
        self.convd_relu_stack = nn.Sequential(
            # 卷积操作以后变为10*24*24
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.ReLU(),
            # 池化以后10*6*6
            nn.MaxPool2d(kernel_size=4),
            # 卷积以后变为：20*4*4
            nn.Conv2d(10, 20, 3),
            nn.ReLU()
        )
        # 全连接层
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20*4*4, 160),
            nn.ReLU(),
            nn.Linear(160,20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        batch_size = x.size(0) # 获取batch_size值
        convd_result = self.convd_relu_stack(x)
        # 将结果的(64,20,4,4)压平成(64,20*4*4),输入到全连接层中
        convd_result = convd_result.view(batch_size,-1)
        result_ts = self.linear_relu_stack(convd_result)
        return result_ts
#-------------------------------------------------------------------------------------------------------
from torch.utils.data import random_split

train_size = int(0.8*len(drDataset))
test_size = int(0.2*len(drDataset))
train_dataset,test_dataset = random_split(drDataset,[train_size,test_size])  # 训练集：测试集≈8:2

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
#-------------------------------------------------------------------------------------------------------
n_epochs = 10 # 迭代次数
learn_rate = 0.001 # 学习率
size = test_size + train_size
device = 'cuda' if torch.cuda.is_available() else "cpu"
model = CNN().to(device)
import torch.optim as optim
# 定义交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
# 定义adam优化器
optimizer = optim.Adam(model.parameters(), lr=learn_rate)
def train_loop(dataloader,model,loss_fn, optimizer):
    for n,(y,x) in enumerate(dataloader):
        # 注意要和权重的类型保持相同
        x = x.float().to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred,y)
        # 存储的变量梯度清零
        optimizer.zero_grad()
        # 求反向传播的梯度
        loss.backward()
        # 开始优化权重
        optimizer.step()
        # 共33600条数据，每进行100个batch输出一次值
        if n%100==0:
            loss,current = loss.item(), (n + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    test_loss, corrent = 0, 0
    size = len(dataloader.dataset)
    batchNum =  len(dataloader)
    with torch.no_grad():
        for y,x in dataloader:
            x = x.float().to(device)
            y = y.to(device)
            pred_y = model(x)
            test_loss += loss_fn(pred_y, y).item()
            # argmax(1)将独热编码的形式解释成标签(0,1,2,3..)最初的形式，type()是为了将bool类型的true转为1，false转为0，这样可以使corrent来计算出预测正确的个数
            corrent += (pred_y.argmax(1)==y).type(torch.float).sum().item()
    # 平均损失,总共的损失除以batch的个数
    arg_loss = test_loss/batchNum
    # 准确率
    correct_rate = corrent/size
    print(f"Test Describe:\nAccuracy: {(100*correct_rate):>0.1f}%, Avg loss: {arg_loss:>8f} \n")

# 对模型进行训练
for n in range(n_epochs):
    print(f"======{n}======")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    print("Done!")

