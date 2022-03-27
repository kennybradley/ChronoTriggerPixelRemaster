import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pickle

#in the background of the image set the color to white
def cleanImage(image):
    if image.shape[2] == 3:
        return image
    img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j][3] == 0:
                img[i,j] = [255,255,255]
            else:
                img[i,j] = image[i,j][:3]

    return img

#Return MSE of difference between two images
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err



root = "."#path redacted
root1 = "FinalFantasy"
root2 = "Remaster"

folders = ["4", "5","6"]

#for each of the folders we are pairing up
for f in folders:
    img1Paths = []
    img2Paths = []

    path1 = root + "FF" + f + root2 + os.sep
    path2 = root + root1 + f + os.sep 

    imageNames1 = os.listdir(path1 + "Cleaned")
    imageNames2 = os.listdir(path2 + "Cleaned")
    #lists to store all the image data
    images1 = [] 
    images2 = [] 
    #read and save the images, save the path names
    for name in sorted(imageNames1):
        img = cv2.imread(path1 + "Cleaned" + os.sep + name, cv2.IMREAD_UNCHANGED)
        img = cleanImage(img)
        images1.append(img)
        img1Paths.append(path1 + "Cleaned" + os.sep + name)
    for name in sorted(imageNames2):
        img = cv2.imread(path2 + "Cleaned" + os.sep + name, cv2.IMREAD_UNCHANGED)
        img = cleanImage(img)
        images2.append(img)
        img2Paths.append(path2 + "Cleaned" + os.sep + name)


    
    #find the max Height and Width of the images
    maxW = 0
    maxH = 0
    for img1 in images1:
        if img1.shape[0] > maxH:
            maxH = img1.shape[0]
        if img1.shape[1] > maxW:
            maxW = img1.shape[1]
    #folder 2 has double sized images, 
    # they are interpolated as nearest so divide down to the proper size
    for img2 in images2:
        if img2.shape[0]//2 > maxH:
            maxH = img2.shape[0]
        if img2.shape[1]//2 > maxW:
            maxW = img2.shape[1]

    #use this as a save state so it can be stopped and restarted
    matchedFiles = []
    if os.path.exists(path2+"matched.pkl"):
        matchedFiles = pickle.load(open(path2+"matched.pkl", "rb"))


    for img1Count, img1 in enumerate(images1):
        #this is to allow for restarting
        if img1Count < len(matchedFiles):
            continue

        mR_minDistance=9999999
        mRIndex=-1
        
        #if image is too large or small skip it
        if img1.shape[1] > 30 or img1.shape[1] < 10 or img1.shape[0] < 10:
            matchedFiles.append(("Too big or small to match", "Too big or small to match"))
            continue

        #array to store the results for each comparison
        comparisonScores = [] 
        for index, img2 in enumerate(images2):
            #the images from the original folder are double sized, resize down with nearest neighbor 
            #since it was upscaled with inter_nearest this should add 0 noise
            img2 = cv2.resize(img2, (img2.shape[1]//2, img2.shape[0]//2), interpolation=cv2.INTER_NEAREST)

            #if the shape is out of range skip the image
            if img2.shape[1] > 30 or img2.shape[1] < 10 or img2.shape[0] < 10:
                continue

            #resize the crop to match the target which should be a minor change at most
            img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)

            #get the difference between the current img2_resized and the target image img1
            mR = mse(img1, img2_resized)
            comparisonScores.append((mR, index))

        #sort by closeness between the images
        comparisonScores.sort()
  
        #grab the top 10
        top10 = comparisonScores[:10]

        #create a background and display each of the top images on that background along
        #with the text of the number it corresponds to
        backdrop = np.ones((250,550,3), dtype=np.uint8)*255
        for i in range(min(len(top10),10)):
            offsetX = (i%5)*100
            offsetY = (i//5)*100
            image = images2[top10[i][1]] 
            image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation=cv2.INTER_NEAREST)
            image = cv2.resize(image, (image.shape[1]*3, image.shape[0]*3), interpolation=cv2.INTER_NEAREST)
            offsetXStop = offsetX+image.shape[1]
            offsetYStop = offsetY+image.shape[0]
            backdrop[offsetY:offsetYStop, offsetX:offsetXStop, :] = image
            backdrop = cv2.putText(backdrop, str(i), (offsetX+50,offsetY+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)

        #triple the X and Y size to display the images to screen
        #display the remastered img1 and the options that are close to it
        img1_resized_final = cv2.resize(img1, (img1.shape[1]*3, img1.shape[0]*3), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("1", img1_resized_final)
        cv2.imshow("2", backdrop)
        #get the numbered input from the user of which is closest
        key = cv2.waitKey(0)
        keyVal = int(key)
        cv2.destroyAllWindows()

        #if the value is non-numeric that means there was no match found
        if keyVal < 48 or keyVal > 57:
            matchedFiles.append(("No Match Found", "No Match Found"))
        #otherwise save out the path names that count as paired
        else:
            matchedFiles.append((img1Paths[img1Count], img2Paths[top10[keyVal-48][1]]))

        #dump out the updated pairings each step 
        #this is overkill but was useful for quick restarting since the process could take a while
        pickle.dump(matchedFiles, open(path2+"matched.pkl", "wb"))
