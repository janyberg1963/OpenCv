import cv2
import numpy as np
import matplotlib.pyplot as plt


import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Read reference image
refFilename ="form.jpg"
print("Reading reference image : ", refFilename)
imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

plt.imshow(imReference[:,:,::-1])
plt.show()

# Read image to be aligned
imFilename = "scanned-form.jpg"
print("Reading image to align : ", imFilename);  
im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

plt.imshow(im[:,:,::-1])
plt.show()


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

# Convert images to grayscale
im1Gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im2Gray = cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors.
orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]

imMatches = cv2.drawMatches(im, keypoints1, imReference, keypoints2, matches, None)
cv2.imwrite("matches.jpg", imMatches)

plt.imshow(imMatches[:,:,::-1])
plt.show()


# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt


h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)   

# Use homography
height, width, channels = imReference.shape
im1Reg = cv2.warpPerspective(im, h, (width, height))

plt.imshow(im1Reg[:,:,::-1])
plt.show()

# Print estimated homography
print("Estimated homography : \n",  h)