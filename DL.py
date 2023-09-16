import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./DL.jpg',0)
for i in range(2000):
    temp_x = np.random.randint(0,img.shape[0])
    temp_y = np.random.randint(0, img.shape[1])
    img[temp_x][temp_y] = 255

blur_1 = cv2.GaussianBlur(img,(5,5),0)

blur_2 = cv2.medianBlur(img,5)

plt.subplot(1,3,1),plt.imshow(img,'gray')
plt.subplot(1,3,2),plt.imshow(blur_1,'gray')
plt.subplot(1,3,3),plt.imshow(blur_2,'gray')
plt.show()
