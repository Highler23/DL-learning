import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms
from torchvision import datasets
import torch.nn.functional as F

# 超参数设置
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))
train_data_dir = "F:/data-for-deep-learning/cat_dog_2000/train/"
test_data_dir = "F:/data-for-deep-learning/cat_dog_2000/test/"
train_batch_size = 4
test_batch_size = 4

# 图像加载与预处理
train_transforms = torchvision.transforms.Compose({
    torchvision.transforms.Resize([256,256]),
    torchvision.transforms.CenterCrop([224,224]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # TODO:归一化处理时，如何确定各个通道图像的均值和标准差的取值？
})
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224,224]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # TODO:接着上面的问题，训练个测试用的均值和标准差地值相同吗？
])
train_dataset = datasets.ImageFolder(root=train_data_dir,transform=train_transforms)
train_loader = DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True,num_workers=0)
test_dataset = datasets.ImageFolder(root=test_data_dir,transform=test_transforms)
test_loader = DataLoader(test_dataset,batch_size=test_batch_size,shuffle=True,num_workers=0)
# for X, y in test_loader:
#     print("Shape of X [N, C, H, W]: ", X.shape)
#     print("Shape of y: ", y.shape, y.dtype)
#     break

# 定义模型
class LeNet(nn.Module):
    # 一般在__init__中定义网络需要的操作算子，比如卷积、全连接算子等等
    def __init__(self):
        super(LeNet, self).__init__()
        # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 由于上一层有16个channel输出，每个feature map大小为53*53，所以全连接层的输入是16*53*53
        self.fc1 = nn.Linear(16*53*53, 120)
        # TODO:这些全连接层为什么不一次性(16*53*53,10)?
        self.fc2 = nn.Linear(120, 84)
        # 最终有10类，所以最后一个全连接层输出数量是10
        self.fc3 = nn.Linear(84,10)
        self.pool = nn.MaxPool2d(2, 2)
    # forward这个函数定义了前向传播的运算，只需要像写普通的python算数运算那样就可以了
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 下面这步把二维特征图变为一维，这样全连接层才能处理
        x = x.view(-1, 16*53*53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # TODO:这种写法标准吗？
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:  # TODO:这里为什么不像上面那样写？for batch, (X, y) in enumerate(dataloader)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")

# TODO:为什么第一次运行，会报错：TypeError: img should be Tensor Image. Got <class 'PIL.Image.Image'>
#      但是终止运行后再次运行就可以正常运行？