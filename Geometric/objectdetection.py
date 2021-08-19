
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0,10.0)
matplotlib.rcParams['image.cmap'] = 'gray'


def createDir(folder):
    try:
      os.makedirs(folder)
    except OSError:
       print('{}: already exists'.format(folder))
    except Exception as e:
      print(e)




def getImagePaths(folder, imgExts):
    imagePaths = []
    for x in os.listdir(folder):
      xPath = os.path.join(folder, x)
    if os.path.splitext(xPath)[1] in imgExts:
      imagePaths.append(xPath)
    return imagePaths


def getDataset(folder, classLabel):
      images = []
      labels = []
      imagePaths = getImagePaths(folder, ['.jpg', '.png', '.jpeg'])
      for imagePath in imagePaths:
    # print(imagePath)
        im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        images.append(im)
        labels.append(classLabel)
        return images, labels

def svmInit(C, gamma):
      model = cv2.ml.SVM_create()
      model.setGamma(gamma)
      model.setC(C)
      model.setKernel(cv2.ml.SVM_LINEAR)
      model.setType(cv2.ml.SVM_C_SVC)
      model.setTermCriteria((cv2.TERM_CRITERIA_EPS + 
                                cv2.TERM_CRITERIA_MAX_ITER, 
                                1000, 1e-3))
      return model


def svmTrain(model, samples, labels):
  model.train(samples, cv2.ml.ROW_SAMPLE, labels)

def svmPredict(model, samples):
  return model.predict(samples)[1]



def svmEvaluate(model, samples, labels):
      labels = labels[:, np.newaxis]
      pred = model.predict(samples)[1]
      correct = np.sum((labels == pred))
      err = (labels != pred).mean()
      print('label -- 1:{}, -1:{}'.format(np.sum(pred == 1), 
          np.sum(pred == -1)))
      return correct, err * 100


def createDir(folder):
      try:
        os.makedirs(folder)
      except OSError:
        print('{}: already exists'.format(folder))
      except Exception as e:
        print(e)


def computeHOG(hog, images):
  hogFeatures = []
  for image in images:
       hogFeature = hog.compute(image)
       hogFeatures.append(hogFeature)
  return hogFeatures



def prepareData(hogFeatures):
   featureVectorLength = len(hogFeatures[0])
   data = np.float32(hogFeatures).reshape(-1, featureVectorLength)
   return data



winSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = -1
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = True
nlevels = 64
signedGradient = False

# Initialize HOG
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                      cellSize, nbins,derivAperture,
                      winSigma, histogramNormType, L2HysThreshold, 
                      gammaCorrection, nlevels,signedGradient)

winSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = -1
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = True
nlevels = 64
signedGradient = False

# Initialize HOG
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                      cellSize, nbins,derivAperture,
                      winSigma, histogramNormType, L2HysThreshold, 
                      gammaCorrection, nlevels,signedGradient)


# Flags to turn on/off training or testing
trainModel = True
testModel = True
queryModel = True


# Path to INRIA Person dataset
rootDir = '/INRIAPerson/'

# set Train and Test directory paths
trainDir = os.path.join(rootDir, 'train_64x128_H96')
testDir = os.path.join(rootDir, 'test_64x128_H96')



if trainModel:
    # Read images from Pos and Neg directories
    trainPosDir = os.path.join(trainDir, 'posPatches')
    trainNegDir = os.path.join(trainDir, 'negPatches')

    # Label 1 for positive images and -1 for negative images
    trainPosImages, trainPosLabels = getDataset(trainPosDir, 1)
    trainNegImages, trainNegLabels = getDataset(trainNegDir, -1)

    # Check whether size of all positive and negative images is same
    print(set([x.shape for x in trainPosImages]))
    print(set([x.shape for x in trainNegImages]))

    # Print total number of positive and negative examples
    print('positive - {}, {} || negative - {}, {}'
        .format(len(trainPosImages),len(trainPosLabels),
        len(trainNegImages),len(trainNegLabels)))

    # Append Positive/Negative Images/Labels for Training
    trainImages = np.concatenate((np.array(trainPosImages), 
                       np.array(trainNegImages)), 
                                  axis=0)
    trainLabels = np.concatenate((np.array(trainPosLabels), 
                       np.array(trainNegLabels)),
                                  axis=0)
    # Now compute HOG features for training images and 
    # convert HOG descriptors to data format recognized by SVM.
    # Compute HOG features for images
    hogTrain = computeHOG(hog, trainImages)

    # Convert hog features into data format recognized by SVM
    trainData = prepareData(hogTrain)

    # Check dimensions of data and labels
    print('trainData: {}, trainLabels:{}'
            .format(trainData.shape, trainLabels.shape))
    # Finally create an SVM object, train the model and save it.
    # Initialize SVM object
    model = svmInit(C=0.01, gamma=0)
    svmTrain(model, trainData, trainLabels)
    model.save('pedestrian.yml')