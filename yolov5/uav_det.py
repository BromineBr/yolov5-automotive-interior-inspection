import os
# 换成无人机的流即可使用。
# 辣椒的检测作为暂时的模板，添加了对应的检测接口
os.system("python detect.py --weights runs/train/exp_transformer/weights/best.pt --source 0")