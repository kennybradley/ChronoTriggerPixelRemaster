import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import cv2
import os

targetX = 48
targetY = 48
trainSize = 1.0

#check for foreground vs background using transparancy
#set all background to black
def cleanImage(image):
    if image.shape[2] == 3:
        return image
        
    img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j][3] == 0:
                img[i,j] = [0,0,0]
            else:
                img[i,j] = image[i,j][:3]
    return img


#load the image, resize if necessary, paste on blank canvas and return as uint8
def readAndConvert(path, resize=False, show=False):
    if path.find(".") == -1:
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if resize:
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_NEAREST)

    if img.shape[0] > targetX or img.shape[1] > targetY:
        return None

    img = cleanImage(img)
    
    yOffset = (targetY - img.shape[0])//2
    xOffset = (targetX - img.shape[1])//2

    adjusted = np.zeros((targetX, targetY, 3), dtype=np.uint8)
    adjusted[yOffset:yOffset+img.shape[0], xOffset:xOffset+img.shape[1],:] = img
    return np.array(adjusted, dtype=np.uint8)


#load the paired dataset between Original and Pixel Remastered
def loadData_XY(picklePath):
    matchedData = pickle.load(open(picklePath, "rb"))
    x_train = []
    y_train = []

    for x,y in matchedData:
        #read and convert the images
        xval = readAndConvert(x)
        yval = readAndConvert(y, True)
        if xval is None or yval is None:
            continue

        #add the images as they are
        x_train.append(xval)
        #add the images rotated along the X axis
        x_train.append(cv2.flip(xval, 1))

                
        #add the images as they are
        y_train.append(yval)
        #add the images rotated along the X axis
        y_train.append(cv2.flip(yval, 1))

        #stretch the examples bigger than they start
        for stretch in [1.1,1.2,1.3,1.4]:

            #increase the size of the crops
            xval_adj = cv2.resize(xval, (int(xval.shape[1]*stretch), int(xval.shape[0]*stretch)), interpolation=cv2.INTER_NEAREST)
            yOffset = (targetY - xval_adj.shape[0])//2
            xOffset = (targetX - xval_adj.shape[1])//2

            if yOffset < 0:
                xval_adj = xval_adj[-yOffset:yOffset, -yOffset:yOffset]
                yOffset=0
                xOffset=0

            #create a blank canvas
            Xadjusted = np.zeros((targetX, targetY, 3), dtype=np.uint8)
            #insert the crop onto the blank canvas
            Xadjusted[yOffset:yOffset+xval_adj.shape[0], xOffset:xOffset+xval_adj.shape[1],:] = xval_adj

            #insert the resized image
            x_train.append(Xadjusted)
            #and again with flipped X axis
            x_train.append(cv2.flip(Xadjusted, 1))

            #Repeat the process with the Y values
            yval_adj = cv2.resize(yval, (int(yval.shape[1]*stretch), int(yval.shape[0]*stretch)), interpolation=cv2.INTER_NEAREST)

            yOffset = (targetY - yval_adj.shape[0])//2
            xOffset = (targetX - yval_adj.shape[1])//2
            if yOffset < 0:
                yval_adj = yval_adj[-yOffset:yOffset, -yOffset:yOffset]
                yOffset=0
                xOffset=0

            #create a blank canvas
            Yadjusted = np.zeros((targetX, targetY, 3), dtype=np.uint8)
            #insert the crop onto the blank canvas
            Yadjusted[yOffset:yOffset+xval_adj.shape[0], xOffset:xOffset+xval_adj.shape[1],:] = yval_adj


            addY = Yadjusted
            addY_flip = cv2.flip(Yadjusted, 1)
            
            #insert the resized image
            y_train.append(addY)
            #and again with flipped X axis
            y_train.append(addY_flip)

    #convert to numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train

#load the image and convert it to the cleaned format we want
def loadData_X(folderPath):
    images =[]
    for root, folders, files in os.walk(folderPath):
        for f in files:
            img = readAndConvert(os.path.join(root,f), resize=True, show=False)
            if img is None:
                continue
            images.append(img)
    return np.array(images)


#load in the image pairings for FF5 and FF6
x_train, y_train = loadData_XY(".\\FF5Remaster\\matched.pkl")#path redacted
x_train2, y_train2 = loadData_XY(".\\FF6Remaster\\matched.pkl")#path redacted

#merge the values from the two folders
x_train = np.concatenate((x_train, x_train2), axis=0)
y_train = np.concatenate((y_train, y_train2), axis=0)

#create the training data and the test data according to the training size at the top
trainA = x_train[:int(np.round(len(x_train)*trainSize))]
trainB = y_train[:int(np.round(len(y_train)*trainSize))]
testA = x_train[int(np.round(len(x_train)*trainSize)):]
testB = y_train[int(np.round(len(y_train)*trainSize)):]

#create a dict to dump the subsets into
dump = {}
dump["trainA"] = trainA
dump["trainB"] = trainB
dump["testA"] = testA
dump["testB"] = testB

#save out the pixel data
pickle.dump(dump, open("pixelart_dataset.pkl", "wb"))

#Load the non-matched and dump the results
CTData = loadData_X(".\\ChronoTrigger\\Cleaned\\")#path redacted
pickle.dump(CTData, open("CT.pkl", "wb"))
