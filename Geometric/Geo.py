import cv2

import numpy as np

import matplotlib.pyplot as plt

import matplotlib
from numpy.core.fromnumeric import size


im_src = cv2.imread(r'Geometric\book2.jpg')
# Four corners of the book in source image
pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]], dtype=float)


# Read destination image.

im_dst = cv2.imread(r"Geometric\book1.jpg")
# Four corners of the book in destination image.
pts_dst = np.array([[0,0],[299,0],[299,399],[0,399]], dtype=float)

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h,(300,400))

# Display images 
plt.figure(figsize=[20,10])
plt.subplot(131)
plt.imshow(im_src[...,::-1])

plt.subplot(132)
plt.imshow(im_dst[...,::-1])

plt.subplot(133)
plt.imshow(im_out[...,::-1])

plt.waitforbuttonpress()