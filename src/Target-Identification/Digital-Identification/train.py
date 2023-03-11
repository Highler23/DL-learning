import momentum
import torch
import torchvision
from torch import optim
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 定义超参数
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
EPOCH = 1
data_path = './data/'

# 准备数据集 这里我们使用MINST数据集
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # Normalize()转换使用的值0.1307和0.3081是MNIST数据集的全局平均值和标准偏差
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
train_data = torchvision.datasets.MNIST(data_path,train=True,transform=transform,download=True)
test_data = torchvision.datasets.MNIST(data_path,train=False,transform=transform,download=True)
train_loader = DataLoader(train_data,batch_size=batch_size_train,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size_test,shuffle=True)

# 构建网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # The size of the picture is 28x28
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 14x14
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 7x7
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output

net = Net()
loss_fn = torch.nn.CrossEntropyLoss()  # 对于简单的多分类任务，我们可以使用交叉熵损失来作为损失函数
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

# 模型训练
print("using device: {}".format(device))  # 查看当前使用设备
history = {'Test Loss': [], 'Test Accuracy': []}  # 存储训练过程  TODO:这里为啥可以这么写
for epoch in range(1, EPOCH + 1):
    processBar = tqdm(train_loader, unit='step')
    net.train(True)
    for step, (trainImgs, labels) in enumerate(processBar):  # TODO:这里的step是哪里的？
        # 这一句的enumerate(processBar)给(trainImgs, labels)返回了什么？它给step返回了索引吗，比如0,1,2...?
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)
        # 简便写法：trainImgs,labels = trainImgs.to(device),labels.to(device)

        net.zero_grad()  # 梯度清零  # TODO:这种清零方式和使用optimizer.step()优化器清零方式有区别吗？
        outputs = net(trainImgs)  # 前向传播
        loss = loss_fn(outputs, labels)  # 计算损失值
        predictions = torch.argmax(outputs, dim=1)  # TODO: 不是很懂这个函数argmax，这里的意思是返回预测结果中最大概率的元素所在列吗？
        accuracy = torch.sum(predictions == labels) / labels.shape[0]  # 计算正确律# TODO：不懂这句为啥这么写
        loss.backward()  # 将损失loss 向输入侧进行反向传播；同时对需要进行梯度计算的变量(requires_grad=True)计算梯度并将其累积到梯度
        optimizer.step()  # 优化器对变量的值进行更新
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % (epoch, EPOCH, loss.item(), accuracy.item()))

        if step == len(processBar) - 1:  # TODO：这句也不懂，为什么拿进度条做判断条件
            correct, totalLoss = 0, 0
            net.train(False)  # 关闭模型的训练状态
            for testImgs, labels in test_loader:
                testImgs = testImgs.to(device)
                labels = labels.to(device)

                outputs = net(testImgs)
                loss = loss_fn(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                totalLoss += loss
                correct += torch.sum(predictions == labels)

            testAccuracy = correct / (batch_size_test * len(test_loader))  # 计算总测试的平均准确率
            testLoss = totalLoss / len(test_loader)  # 计算总测试的平均Loss
            history['Test Loss'].append(testLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                       (epoch, EPOCH, loss.item(), accuracy.item(), testLoss.item(),
                                        testAccuracy.item()))
    processBar.close()

torch.save(net,'./result/model.pth')  # 注意这里不能自动创建目录，所以保存前先确认目录是否已经创建

# TODO:一些问题
#       1.验证集一般都是由训练集按8：2比例分出来的吗？
#       2.一般是在训练代码中将使用验证集吗？
#       3.既然已经在训练代码中更新了梯度，并进行了优化，为什么还要写验证代码？为什么要多次计算准确律等参数？
#       4.标准的模型训练，测试代码怎么写？一些细节比如：超参数命名规则、变量的命名规则......
#       5.需要分别定义训练函数和验证函数吗？比如这一篇：https://blog.csdn.net/qq_45550375/article/details/126446155
#       6.模块化编程在深度学习中有必要吗？
#       7.数据集一般采用哪种格式？不同模型需要的不同格式的数据集任何选择？
#
#
