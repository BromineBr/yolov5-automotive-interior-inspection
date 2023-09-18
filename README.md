# 基于深度学习的汽车内饰出厂检测demo



## 技术方案

1. 利用YOLOv5训练目标检测模型（方向盘、副驾驶前盖板），对图片/视频抽帧进行目标检测及Box crop。

2. 对crop的部分转化为RGB格式，再转化成数组后进行K-Means聚类，从而确定该区域内的主要颜色。

3. 将该颜色同正确颜色做对比，输出结果。




## 项目部署

### 环境配置

创建虚拟环境

```
conda create -n yolo python==3.8.5
conda activate yolo
```



PyTorch安装

```
conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 # 30系列以上显卡gpu版本pytorch安装指令
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 10系和20系以及mx系列的执行这条
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # CPU直接执行这条命令即可
```



安装依赖

```
pip install -r requirements.txt
```



### 训练模型

安装labelimg

```
pip install labelimg
```



运行labelimg，对图片打标签

```
labelimg
```

注意修改label保存路径，同时，**输出格式应当改为YOLO格式**。



修改配置文件`my_data.yaml`

```
train: ./datasets/images/train/  #训练集地址
val: ./datasets/images/test/  #测试集、验证集地址

nc: 2  #类的数目

names: ['wheel', 'panel']  #类名
```



运行指令

```
python train.py --data my_data.yaml --cfg yolov5s.yaml --weights pretrained/yolov5s.pt --epoch 50 --batch-size 4
```

结果将保存在`./runs/train`中



### 使用已有模型进行预测

修改`window_main.py`的`line 43`

```
self.model = self.model_load(weights="runs/train/exp9/weights/best.pt",
```

改为想要使用的模型，weights文件夹中的`best.pt`和`last.pt`分别为训练中的最佳模型和最后一个epoch的模型



执行`window_main.py`



### 注意事项

若读图有问题，请将pillow版本降至8.4.0

```
pip install pillow==8.4.0
```



若numpy报错`module 'numpy' has no attribute 'int'`，请降低numpy的版本



## 问题及改进方向

- 因为数据集过小，并且数据集中的照片车型不同，光线和色温也有差异，因此导致检测准确率低。

- 实际使用中光线固定和色温固定、车型固定，且数据量更大，跑出来的模型会更准确。
- 可添加纠错功能，在初期使用过程中如果出错，可以人工矫正，这可以进一步训练模型，使模型在使用中更加准确。



## 注意事项

- 实际模型训练过程中，应注意从多角度拍摄照片。

- 实际拍摄过程中，应注意虚化问题。



## 引用

1. Ultralytics, yolov5, https://github.com/ultralytics/yolov5
2. 肆十二, YOLOV5-sfid, https://gitee.com/song-laogou/yolov5-sfid
3. 数据集使用：BEHNAM HASANBEYGI, Persian Car Interior Design, https://www.kaggle.com/datasets/behnamhasanbeygi/persian-car-interior-design



![](https://github.com/BromineBr/yolov5-automotive-interior-inspection/blob/main/images/UI/waifu.png)

Produced by Bromine_Br & Hina Kagiyama
