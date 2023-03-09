# yolov5参数解析

***

## detect.py

有默认值参数

* **weights** 权重文件
* **source** 待检测图片/视频/...的路径
* **img-size** 训练过程中缩放图像尺寸，一般为640，1280；不会改变输入和输出
* **conf-thres** 执行度，只有当识别的物体概率大于设置的默认值时才会在输出图上标记出来
* **iou-thres** 同一物体出现多个框选时，按照iou计算公式(相交面积除以并集面积)，若结果大于阈值则使用一个框标记，即视为同一个物体，否则不进行处理
* **device**  cuda 

无默认值参数(一旦设置，将赋值为true)

* **view-img**  弹窗显示检测数据
* **save-txt**  保存结果的标签
* **nosave** 不保存图片和视频
* **classes**  通过索引指定相应的检测对象    比如：`--classes 0`只检测人类
* **agnostic-nms**  增强nms
* **augment**  增强检测能力
* **update**  只保留与预测有关的功能
* **project**  结果的保存路径
* **name**  保存在project指定路径下的文件名/文件夹名
* **exist-ok**  若为true，则新的结果将保存在上一个结果的路径下

所有的参数值最终都会保存到opt变量中

***

## train.py

* **weight**  权重文件，注意在训练时将默认值设为空，这样才能通过训练调整参数
* **cfg**   模型结构
* **data**  训练数据集
* **hyp**   超参数   Q：什么是超参数
* **epochs**   训练轮数
* **batch-size**  把多少数据打包成一个batch
* **img-size**   训练图像尺寸
* **rect**  矩阵训练方式   减少多余的填充，加快训练速度
* **resume**   指定一个基础，即在指定模型的基础上进行自己的训练 这个模型是之前训练输出的模型文件，到相应路径下寻找即可
* **nosave**    只保存最后一轮训练输出的模型的权重数据，`.pt`文件
* **notest**  是否只在最后一轮训练上测试
* **noautoanchor**  锚点，锚框
* **evolve**  默认为true 对参数净化 是寻找最优参数的方式之一
* **bucket**  一般不使用
* **cache-image**  是否对图片缓存
* **image-weights**   对上一轮训练中部分效果不好的数据在下一论训练中添加相关权重
* **device**  
* **multi-scale**  对图片进行变换
* **single-cls**  训练的数据集是但类别还是多类别
* **adam**  优化器/梯度下降
* **sync-bn**  多GPU训练
* **local-rank**  DDP参数
* **workers**  
* **project**  结果的保存路径
* **entity**
* **name**  保存在project指定路径下的文件名/文件夹名
* **exist-ok**  若为true，则新的结果将保存在上一个结果的路径下
* **quad**  
* **linear-lr**  学习速率会按照线性方式处理，否则按照余弦方式处理
* **label-smoothing**  标签平滑，防止过拟合
* **upload-dataset**
* **bbox-interval**
* **save-period**  对模型打印日志，默认为-1
* **artifact-alias**  (还没用开发出来，也没啥用)

