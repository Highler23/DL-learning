'''
    鐜锛歸indows10 + anaconda4.13.0 + pytorch cuda113 + python3.7
'''
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# 鍑嗗鏁版嵁闆�
class TitanicDataset(Dataset):

    def __init__(self, filepath):
        xy = pd.read_csv(filepath)  # 璇诲叆csv鏂囦欢
        self.len = xy.shape[0]  # xy.shape()鍙互寰楀埌xy鐨勮鍒楁暟;shape[0]琛ㄧず琛屾暟锛宻hape[1]琛ㄧず鍒楁暟
        # 閫夊彇鐩稿叧鐨勬暟鎹壒寰�
        feature = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
        # 鐗瑰緛鎻愬彇
        self.x_data = torch.from_numpy(np.array(pd.get_dummies(xy[feature])))  # np.array()灏嗘暟鎹浆鎹㈡垚鐭╅樀锛屾柟渚胯繘琛屾帴涓嬫潵鐨勮绠�
        self.y_data = torch.from_numpy(np.array(xy["Survived"]))

    # getitem鍑芥暟锛屽彲浠ヤ娇鐢ㄧ储寮曟嬁鍒版暟鎹�
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # 杩斿洖鏁版嵁鐨勬潯鏁�/闀垮害
    def __len__(self):
        return self.len

dataset = TitanicDataset('data/train.csv')  # 瀹炰緥鍖栬嚜瀹氫箟绫伙紝骞朵紶鍏ユ暟鎹湴鍧€
train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)

# 瀹氫箟妯″瀷
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 瑕佸厛瀵归€夋嫨鐨勭壒寰佽繘琛岀嫭鐑〃绀鸿绠楀嚭缁村害锛岃€屽悗鍐嶉€夋嫨绁炵粡缃戠粶寮€濮嬬殑缁村害
        self.linear1 = torch.nn.Linear(6, 3)
        self.linear2 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()

    # 鍓嶉鍑芥暟
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x

    # 娴嬭瘯鍑芥暟
    def test(self, x):
        with torch.no_grad():
            x = self.sigmoid(self.linear1(x))
            x = self.sigmoid(self.linear2(x))
            y = []
            # 鏍规嵁浜屽垎娉曞師鐞嗭紝鍒掑垎y鐨勫€�
            for i in x:
                if i > 0.5:
                    y.append(1)
                else:
                    y.append(0)
            return y

# 瀹炰緥鍖栨ā鍨�
model = Model()

# 瀹氫箟鎹熷け鍑芥暟
criterion = torch.nn.BCELoss(reduction='mean')

# 瀹氫箟浼樺寲鍣�
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    # 閲囩敤澶氬眰宓屽寰幆
    for epoch in range(100):
        # data浠巘rain_loader涓彇鍑烘暟鎹紙鍙栧嚭鐨勬槸涓€涓厓缁勬暟鎹級锛氾紙x锛寉锛�
        # enumerate鍙互鑾峰緱褰撳墠鏄鍑犳杩唬锛屽唴閮ㄨ凯浠ｆ瘡涓€娆¤窇涓€涓狹ini-Batch
        for i, data in enumerate(train_loader, 0):
            # inputs鑾峰彇鍒癲ata涓殑x鐨勫€硷紝labels鑾峰彇鍒癲ata涓殑y鍊�
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

# 娴嬭瘯
test_data = pd.read_csv('data/test.csv')
feature = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
test = torch.from_numpy(np.array(pd.get_dummies(test_data[feature])))
y = model.test(test.float())

# 杈撳嚭棰勬祴缁撴灉
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y})
output.to_csv('exp/csv/my_predict.csv', index=False)

#淇濆瓨璁粌濂界殑妯″瀷
torch.save(model.state_dict(), "exp/pth/result.pt")
