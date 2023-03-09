# pytorch

***

## 0. 目录

* 基础知识
* 



***

## 1.基础知识

### 1.1 张量

在 PyTorch 中，我们使用张量对模型的输入和输出以及模型的参数进行编码。	

默认情况下，张量是在 CPU 上创建的；需要使用 using 方法显式地将张量移动到 GPU 中（在检查 GPU 可用性之后）
张量可以在 GPU 或其他硬件加速器上运行

#### 1.1.1 初始化





#### 1.1.2 属性

张量属性描述它们的形状、数据类型和存储它们的设备。

#### 1.1.3 操作



### 1.2 数据集

#### 1.2.1 加载数据集

```python
from torch.utils.data import Dataset
from torchvision import datasets  # 预加载数据集资源
```

#### 1.2.2 创建自定义数据集

自定义数据集类必须实现三个函数：__init__、__len__和__getitem__

由于类可以起到模板的作用，因此，可以在创建实例的时候，把一些我们认为必须绑定的属性强制填写进去。通过定义一个特殊的`__init__`方法，在创建实例的时候，就把`name`，`score`等属性绑上去：

self`，表示创建的实例本身，因此，在`__init__`方法内部，就可以把各种属性绑定到`self



#### 1.2.3 数据加载器

```python
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
```

### 1.3 变换

```python
from torchvision import transforms
```

### 1.4 创建简易的神经网络

通过子类化来定义我们的神经网络

`torch.nn`命名空间提供了您需要的所有构建块 构建您自己的神经网络。
PyTorch 中的每个模块都对 nn 进行子类化
神经网络本身是一个模块，由其他模块（层）组成。这种嵌套结构允许 轻松构建和管理复杂的架构。

```python
from torch import nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类的方法
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def __len__(self):
        pass
    
    def __getitem(self):
        pass

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

### 1.5 模型层



### 1.6 优化模型参数

首先介绍一些概念：

* **超参数**：可调整的参数，可让您控制模型优化过程。 不同的超参数值会影响模型训练和收敛率
  ps:详见：https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

  * 纪元数：迭代数据集的次数
  * 批量大小：在更新参数之前通过网络传播的数据样本数
  * 学习率：每个批次/纪元更新模型参数的数量。
    ps:较小的值会导致学习速度变慢，而较大的值可能会导致训练期间不可预测的行为。

* **优化循环**

* 损失函数：测量获得的结果与目标值的差异程度

* **优化**：调整模型参数以减少每个训练步骤中的模型误差的过程

  ps:在训练循环中，优化分三个步骤进行：

  - 调用以重置模型参数的梯度。默认情况下，渐变相加;为了防止重复计算，我们在每次迭代时都明确将其归零。`optimizer.zero_grad()`
  - 通过调用 来反向传播预测损失。PyTorch 存储每个参数的损失梯度。`loss.backward()`
  - 获得梯度后，我们调用通过反向传递中收集的梯度来调整参数。`optimizer.step()`

### 1.7 保存并加载模型

```python
import torchvision.models as models
```

PyTorch 模型将学习的参数存储在内部 状态字典

```python
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
```

要加载模型权重，需要先创建同一模型的实例，然后加载参数 使用方法

## 2. 基础实践















