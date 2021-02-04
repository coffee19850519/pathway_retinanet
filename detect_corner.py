import numpy as np
import cv2
# from matplotlib import pyplot as plt
img = cv2.imread('/mnt/detectron2/pathway_retinanet_weiwei_65k/test/c2.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,20,0.06,10)
# 返回的结果是[[ 311., 250.]] 两层括号的数组。
corners = np.int0(corners)
print(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)


cv2.imwrite('/mnt/detectron2/pathway_retinanet_weiwei_65k/test/c2_ok.jpg', img)