import cv2
import numpy as np
import os

datasetPath = "."#path redacted

for root, folders, files in os.walk(datasetPath):
    #for each Remaster folder (but don't repeat the process within the output/Sprites folders)
    if "Remaster" not in root or "Sprites" in root:
        continue

    for f in files:
        f = root + "\\" + f 
        f.replace("\\", "\\\\")
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            continue

        #make the output path
        if not os.path.exists(root + os.sep + "Sprites"):
            os.mkdir(root + os.sep + "Sprites")
            
        #create a blank space to copy the sprite sheet into        
        imgA = np.zeros((img.shape[0],img.shape[1],4), dtype=np.uint8)

        #add transparency if necessary, change the background to white
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j][0] == 238 and img[i,j][1] == 5 and img[i,j][2] == 255:
                    imgA[i,j]=np.array([255,255,255,0])
                else:       
                    imgA[i,j]=np.array([img[i,j][0], img[i,j][1], img[i,j][2], 255])

        #grab the RGB only
        test= imgA[:,:,:3]
        #convert to gray
        gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        #invert
        grayInv = (255-gray)

        # setting threshold of gray image
        _, threshold = cv2.threshold(grayInv, 127, 255, cv2.THRESH_BINARY)
        
        # using the findContours function to grab individual sprites
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imgCount = len(os.listdir(root + os.sep + "Sprites"))
        
        for contour in contours:
            rect = cv2.boundingRect(contour)
            x,y,w,h = rect

            #very large and tiny crops should be ignored
            if w*h < 10000 and w*h > 10:
                #grab the individual crops
                crop = imgA[y:y+h, x:x+w,:]

                #set a black background with no transparency if this is background
                for i in range(crop.shape[0]):
                    for j in range(crop.shape[1]):
                        if crop[i,j][0] == 255 and crop[i,j][1] == 255 and crop[i,j][2] == 255 and crop[i,j][3] == 0:
                            crop[i,j] = np.array([0,0,0,0])

                #write out the image
                imgCount+=1
                cv2.imwrite(root + os.sep + "Sprites" + os.sep + str(imgCount) + ".png", crop)

