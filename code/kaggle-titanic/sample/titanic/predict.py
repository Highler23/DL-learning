import  torch
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import pandas as pd

dftrain_raw = pd.read_csv('train.csv')
dftest_raw = pd.read_csv('test.csv')

dfresult= pd.DataFrame()
dfPclass = pd.get_dummies(dftest_raw['Pclass']) # 标准化
dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ] # 列名
dfresult = pd.concat([dfresult,dfPclass],axis = 1)
print(dfresult)
dfSex = pd.get_dummies(dftest_raw['Sex']) #性别编码
dfresult = pd.concat([dfresult,dfSex],axis =  1)
dfresult['Age'] = dftest_raw['Age'].fillna(0)
dfresult['Age_null'] = pd.isna(dftest_raw['Age']).astype('int32')

dfresult['SibSp'] =dftest_raw['SibSp']
dfresult['Parch'] = dftest_raw['Parch']
dfresult['Fare'] =dftest_raw['Fare']
dfresult['Cabin_null'] =  pd.isna(dftest_raw['Cabin']).astype('int32')


dfEmbarked = pd.get_dummies(dftest_raw['Embarked'],dummy_na=True)
dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)


x=dfresult
X = torch.from_numpy(x.values).type(torch.FloatTensor)

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
model=torch.load('tatianic.pth')
predict=model(X)
y=predict.detach().numpy()

list1=[]  # 空列表
for i in y:
    if i > 0.5:
        i=1
        list1.append(i)  # 存入列表
    else:
        i=0
        list1.append(i)

list2=[]
for i in range(892,1310):  # python中()是左闭右开的
    list2.append(i)

list=[list1,list2]
names = ['PassengerId','Survived']
test = pd.DataFrame(zip(list2,list1),columns = names)  # 拼接
test.to_csv('predict1.csv',index=False)  # 保存csv文件






