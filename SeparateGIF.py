from PIL import Image
import os

path = "."#internal path redacted
for root, folders, files in os.walk(path):
    if "Chrono" not in root or "Sprites" in root:
        continue

    if not os.path.exists(root + os.sep + "Sprites"):
        os.mkdir(root + os.sep + "Sprites")

    for f in files:
        if f[-3:] == "gif":

            imageObject = Image.open(root + os.sep + f)
            a = imageObject.n_frames

            for frame in range(0,imageObject.n_frames):
                imageObject.seek(frame)
                imageObject.save(root + os.sep + "Sprites"  + os.sep + f[:-4] + "_" + str(frame+1) + ".png", **imageObject.info)
                
 
