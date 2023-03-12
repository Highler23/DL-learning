import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms
from torchvision import datasets

# 超参数设置
data_root = "F:/data-for-deep-learning/cat_dog_2000/train/"
train_batch_size = 64

# 图像加载与预处理
image_preprocess = torchvision.transforms.Compose({
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # TODO:归一化处理时，如何确定各个通道图像的均值和标准差的取值？
})
train_dataset = datasets.ImageFolder(root=data_root,transform=image_preprocess)
train_loader = DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)


