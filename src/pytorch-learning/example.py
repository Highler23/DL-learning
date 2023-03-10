import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd


# 准备数据集
class TitanicDataset(Dataset):
    def __init__(self, filepath):
        xy = pd.read_csv(filepath)
        # xy.shape（）可以得到xy的行列数
        self.len = xy.shape[0]
        # 选取相关的数据特征
        feature = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
        # np.array()将数据转换成矩阵，方便进行接下来的计算
        # 要先进行独热表示，然后转化成array，最后再转换成矩阵
        self.x_data = torch.from_numpy(np.array(pd.get_dummies(xy[feature])))
        self.y_data = torch.from_numpy(np.array(xy["Survived"]))

    # getitem函数，可以使用索引拿到数据
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # 返回数据的条数/长度
    def __len__(self):
        return self.len


# 实例化自定义类，并传入数据地址
dataset = TitanicDataset('train.csv')
# num_workers是否要进行多线程服务，num_worker=2 就是2个进程并行运行
# 采用Mini-Batch的训练方法
train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)


# 定义模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 要先对选择的特征进行独热表示计算出维度，而后再选择神经网络开始的维度
        self.linear1 = torch.nn.Linear(6, 3)
        self.linear2 = torch.nn.Linear(3, 1)

        self.sigmoid = torch.nn.Sigmoid()

    # 前馈
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))

        return x

    # 测试函数
    def test(self, x):
        with torch.no_grad():
            x = self.sigmoid(self.linear1(x))
            x = self.sigmoid(self.linear2(x))
            y = []
            # 根据二分法原理，划分y的值
            for i in x:
                if i > 0.5:
                    y.append(1)
                else:
                    y.append(0)
            return y


# 实例化模型
model = Model()

# 定义损失函数
criterion = torch.nn.BCELoss(reduction='mean')
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 防止windows系统报错
if __name__ == '__main__':
    # 采用Mini-Batch的方法训练要采用多层嵌套循环
    # 所有数据都跑100遍
    for epoch in range(100):
        # data从train_loader中取出数据（取出的是一个元组数据）：（x，y）
        # enumerate可以获得当前是第几次迭代，内部迭代每一次跑一个Mini-Batch
        for i, data in enumerate(train_loader, 0):
            # inputs获取到data中的x的值，labels获取到data中的y值
            x, y = data
            x = x.float()
            y = y.float()
            y_pred = model(x)
            y_pred = y_pred.squeeze(-1)
            loss = criterion(y_pred, y)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# 测试
test_data = pd.read_csv('test.csv')
feature = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
test = torch.from_numpy(np.array(pd.get_dummies(test_data[feature])))
y = model.test(test.float())

# 输出预测结果
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y})
output.to_csv('my_predict.csv', index=False)




