import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torch.autograd import Variable

import pandas as pd
import numpy as np

test = pd.read_csv('../data/test.csv')
test_data = torch.FloatTensor(np.array(test).reshape((-1,1,28,28)))/255

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

test_model = torch.load('../result/model_weights.pth')
# print(test_model)
test_model.eval()
with torch.no_grad():
    output = test_model(test_data[:])
    # pred_test = torch.max(output,1)[1].squeeze(1)  # torch.max(input, dim)返回输入张量给定维度上每行最大值以及每个最大值的位置索引
    out = pd.DataFrame(np.array(output), index=range(1, 1 + len(output)), columns=['ImageId', 'Label'])
    out.to_csv('../result/kaggle.csv')
    print('结果已保存至: {}'.format('../result/kaggle.csv'))

print('即将退出程序......')



