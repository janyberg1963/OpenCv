

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
keypoints=[]
grayimages = []
images = []

keypoint=[]
dirName = "scene" 

imagefiles = [r"Geometric/Images/{}/".format(dirName) + f for f in os.listdir (r"Geometric/Images/"+dirName) if f.endswith(".jpg")]
imagefiles.sort()

destination = "{}_result.png".format(dirName)
plt.figure(figsize=[20,15])
i=1
for filename in imagefiles:
    img = cv2.imread(filename)
    images.append(img)

orb = cv2.ORB_create(MAX_FEATURES) 

for img in images:
    Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(Gray, None)
    im1Keypoints = np.array([])
    im1Keypoints = cv2.drawKeypoints(Gray, keypoints, im1Keypoints, color=(0,0,255),flags=0)

    # plt.imshow(im1Keypoints [:,:,::-1])
    # plt.title("Keypoints obtained from the ORB detector")
    # plt.waitforbuttonpress()


    cv2.imwrite("keypoints.jpg", im1Keypoints)
    keypoint.append(im1Keypoints)
    grayimages.append(Gray)
      
for im in keypoint:
    plt.imshow(im[:,:,::-1])
    plt.title("Keypoints obtained from keypoints detector")
    plt.waitforbuttonpress()
   
    

       










