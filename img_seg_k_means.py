import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:\\Users\\shahk\\Documents\\Shah\\Programming\\Machine_Learning\\python_for_microscopists-master\\images\\BSE_Image.jpg')
print(img.shape)
img2 = img.reshape((-1,3))
img2 = np.float32(img2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Clusters
k = 4
attempts =10
ret,label,center=cv2.kmeans(img2,k,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)

res = center[label.flatten()]
print(res.shape)
res2 = res.reshape((img.shape))
print(res2.shape)

cv2.imwrite('segmented.jpg')