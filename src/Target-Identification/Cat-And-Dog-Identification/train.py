import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms

# 图像预处理
image_preprocess = torchvision.transforms.Compose({
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # TODO:归一化处理时，如何确定各个通道图像的均值和标准差的取值？
})

