import imageio
import os
import numpy as np

sourcePath = "../../VIDEO/MARATON/IMG_L/"
destPath = "../../VIDEO/MARATON/MP4_L/a"
framesPerSec = 25

img_array = []

img_index = [int(x.split(".")[0]) for x in os.listdir(sourcePath)] 
img_index = sorted(img_index)
img_path = [str(x) + ".jpg" for x in img_index]

for path in img_path:
    
    img = imageio.imread(sourcePath + path, 'JPEG')
    img_array.append(img)


imageio.mimsave(destPath, img_array, 'GIF', fps=framesPerSec)
