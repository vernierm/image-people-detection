import PIL
from PIL import Image, ImageDraw
import os
import numpy as np
from keras.models import load_model
import numpy as np
import cv2

model_path = '../../models/modelMy20LargeSetWithAugment.h5'
true_path = '../../dataset/test/true'
false_path = '../../dataset/test/false'
boundary = 0.5
model = load_model(model_path)

def sumFalseErrors(outputs):
    sum = 0
    for i in outputs:
        if i > 0.5:
            sum += 1
    return sum

def sumTrueErrors(outputs):
    sum = 0
    for i in outputs:
        if i <= 0.5:
            sum += 1
    return sum

def predictFolder(folder):
    lst = os.listdir(folder)
    imgList = []
    
    for i in range(0, len(lst)):
        img = Image.open(folder + '/' + lst[i])    
        
        data = np.array(img) 
    
        pix = np.array(img)
        pix = pix/255.

        imgList.append(pix)

    outputs = model.predict(np.array(imgList))

    if folder.endswith("true"):
        sum = sumTrueErrors(outputs)
        print "Out of " + str(len(lst)) + " pictures in true test set: " + str(sum) + " ERRORS!"
    elif folder.endswith("false"):
        sum = sumFalseErrors(outputs)
        print "Out of " + str(len(lst)) + " pictures in false test set: " + str(sum) + " ERRORS!"
    

if __name__ == "__main__":
    print "TEST SCORES"
    predictFolder(true_path)
    predictFolder(false_path)
