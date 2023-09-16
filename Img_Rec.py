import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors


# 读取图片并转换为RGB格式
img = cv2.imread('example.jpg')
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 将图像转换为一维数组
rgb_data = rgb_img.reshape((-1, 3))

# 对数据进行预处理和特征提取
kmeans = KMeans(n_clusters=3)
kmeans.fit(rgb_data)

# 取出聚类中心，作为颜色识别结果
colors = np.array(kmeans.cluster_centers_, dtype='int')

# 循环遍历每个颜色，并输出其名称
for color in colors:
    color_name = webcolors.rgb_to_name(color)
    print(color_name)
