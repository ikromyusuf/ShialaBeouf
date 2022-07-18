import matplotlib.pyplot as plt
import numpy as np
import cv2

# def empty(a):
#     pass

# path = 'ShiaLaBeouf_green.jpg'
# cv2.namedWindow('TrackBars')
# cv2.resizeWindow('TrackBars',640,240)
# cv2.createTrackbar('Hue min','TrackBars',0,179,empty)
# cv2.createTrackbar('Hue max','TrackBars',179,179,empty)
# cv2.createTrackbar('Sat min','TrackBars',0,255,empty)
# cv2.createTrackbar('Sat max','TrackBars',255,255,empty)
# cv2.createTrackbar('val min','TrackBars',0,255,empty)
# cv2.createTrackbar('val max','TrackBars',255,255,empty)
# while True:
#     img = cv2.imread(path)
#     imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     h_min = cv2.getTrackbarPos('Hue min','TrackBars')
#     h_max = cv2.getTrackbarPos('Hue max','TrackBars')
#     s_min = cv2.getTrackbarPos('Sat min','TrackBars')
#     s_max = cv2.getTrackbarPos('Sat max','TrackBars')
#     v_min = cv2.getTrackbarPos('val min','TrackBars')
#     v_max = cv2.getTrackbarPos('val max','TrackBars')
#     lower = np.array([h_min,s_min,v_min])
#     upper = np.array([h_max,s_max,v_max])
#     mask = cv2.inRange(imgHSV,lower,upper)
#     cv2.imshow('orginal',img)
#     cv2.imshow('HSV',imgHSV)
#     cv2.imshow('Mask',mask)
#     cv2.waitKey(1)

plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False

def show(img):
  plt.figure(figsize=(5,5),dpi=200)
  plt.imshow(img,cmap='gray')
  # plt.axis('off')
  plt.show()

img = cv2.imread('ShiaLaBeouf_green.jpg')
plt.figure(figsize=(8,8))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


lower = np.array([44,63,142])
upper = np.array([71,252,247])

# gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hum = cv2.inRange(img_HSV,lower,upper)

show(hum)
# plt.imshow(hum,cmap='gray')
# plt.show()




# ROI_img2 = cv2.bitwise_and(img,hum)
# show(ROI_img2)
# print(ROI_img2.shape)