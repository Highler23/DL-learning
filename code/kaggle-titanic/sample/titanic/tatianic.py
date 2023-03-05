import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

dftrain_raw = pd.read_csv('train.csv')
dftest_raw = pd.read_csv('test.csv')

dftrain_raw.head(10)
df1=dftrain_raw.drop(['Name','Ticket'], axis=1)  # 删除这两列数据
df1.head(10)  #查看前十行数据
print(df1.isnull().sum())

dfresult= pd.DataFrame()  # 赋值一个空的表格，dataframe，用于保存数据
# dataframe.value也是numpy类型数据

# 将pclass存入前面空的表格
dfPclass = pd.get_dummies(dftrain_raw['Pclass']) # 标准化  独热编码函数，参数是需要独热编码的数据
dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ] # 列名  给数据编码
dfresult = pd.concat([dfresult,dfPclass],axis = 1)  # 保存数据 axis = 0 按列，1按行
print(dfresult)

dfSex = pd.get_dummies(dftrain_raw['Sex']) #性别编码
dfresult = pd.concat([dfresult,dfSex],axis =  1)

dfresult['Age'] = dftrain_raw['Age'].fillna(0)  # 因为本身为数字，直接将年龄存入
dfresult['Age_null'] = pd.isna(dftrain_raw['Age']).astype('int32')  # 没有年龄数据的填0

dfresult['SibSp'] =dftrain_raw['SibSp']
dfresult['Parch'] = dftrain_raw['Parch']
dfresult['Fare'] = dftrain_raw['Fare']
dfresult['Cabin_null'] =  pd.isna(dftrain_raw['Cabin']).astype('int32')


dfEmbarked = pd.get_dummies(dftrain_raw['Embarked'],dummy_na=True)
dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

print(dfresult)

# 数据类型转换
y= dftrain_raw['Survived'].values
y=np.reshape(y,(891,1))
x=dfresult

X = torch.from_numpy(x.values).type(torch.FloatTensor)
Y = torch.from_numpy(y).type(torch.FloatTensor)


##___________________________________________________________________________________
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.liner_1 = nn.Linear(15, 8)
#         self.liner_2 = nn.Linear(8, 4)
#         self.liner_3 = nn.Linear(4, 1)
#
#     def forward(self, input):
#         x = F.relu(self.liner_1(input))
#         x = F.relu(self.liner_2(x))
#         x = torch.sigmoid(self.liner_3(x))
#         return x

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(15, 12)
        self.mish1 = Mish()
        self.linear2 = nn.Linear(12, 8)
        self.mish2 = Mish()
        self.liner_3 = nn.Linear(8, 1)



    def forward(self, x):
        x = self.linear1(x)
        x = self.mish1(x)
        x = self.linear2(x)
        x = self.mish2(x)
        x = torch.sigmoid(self.liner_3(x))
        return x

model = Model()
lr = 0.0001

opt = torch.optim.Adam(model.parameters(), lr=0.0001)
def get_model():
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt


loss_fn = nn.BCELoss()
batch = 64
epochs = 10000


##------------------------------------------------------------

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y)  # 自动划分为8 2 开，数据集和测试集

train_x=np.array(train_x)  # 数据类型转换
train_y=np.array(train_y)
test_x=np.array(test_x)
test_y=np.array(test_y)

train_x=torch.tensor(train_x).type(torch.float32)
train_y=torch.tensor(train_y).type(torch.float32)
test_x=torch.tensor(test_x).type(torch.float32)
test_y=torch.tensor(test_y).type(torch.float32)

train_ds=TensorDataset(train_x,train_y)  # 拼接数据
train_dl=DataLoader(train_ds,batch_size=batch,shuffle=True)  # 加载数据
# 单个值传递，会引起波动，变化很大，且训练速度慢，所以一般按批次加载数据
test_ds=TensorDataset(test_x,test_y)
test_dl=DataLoader(test_ds,batch_size=batch)

#计算正确率

def accuracy(y_pred,y_true):
    y_pred = (y_pred > 0.5).type(torch.int32)
    acc=(y_pred == y_true).float().mean()
    return acc

# ---------------------------------------------主函数
model, optim = get_model()
for epoch in range(epochs):
    for x, y in train_dl:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        # 反向传播
        optim.zero_grad()  # 优化器  置0
        loss.backward()
        optim.step()  # 优化器更新
    with torch.no_grad():  # 使正确率的计算结果不参与优化更新
        epoch_train_accuracy=accuracy(model(train_x),train_y)
        epoch_test_accuracy=accuracy(model(test_x),test_y)
        print('epoch:', epoch, 'loss:', loss_fn(model(X), Y).data.item(),'train accuracy:',epoch_train_accuracy,
              'test accuracy',epoch_test_accuracy)

torch.save(model,'tatianic.pth')