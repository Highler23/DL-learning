# 使用transform：
#       关注输入输出
#       多看官方文档
#       初始化需要的参数、一般保留默认值、寻找相应的作用
#       关注方法需要什么参数，以及输出什么数据类型，那就print(type())或者打断点查看或者网上查询
# __call__的用法
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("images/pytorch.png")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)


writer.close()





