import numpy as np
import cv2
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Read input image
img = cv2.imread(r"Geometric\book.jpeg")
# Convert to grayscale
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12,12))
plt.subplot(1,2,1)
plt.imshow(img[:,:,::-1])
plt.subplot(1,2,2)
plt.imshow(imgGray)
plt.show()


# Initiate ORB detector
orb = cv2.ORB_create(10)

# find the keypoints with ORB
kp = orb.detect(imgGray,None)

# compute the descriptors with ORB
kp, des = orb.compute(imgGray, kp)
# draw keypoints location, size and orientation
img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(12,12))
plt.imshow(img2[:,:,::-1])
plt.waitforbuttonpress()



