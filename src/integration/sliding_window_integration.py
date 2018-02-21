import numpy as np
from keras.models import load_model
from Tkinter import Tk
from tkFileDialog import askopenfilename
from PIL import Image, ImageDraw
import time


#GLOBAL VARIABLES
modelPath = "../../models/slozenija_20_vise_slika.h5"
output_path = "../../test/img.jpg"

#SLIDING WINDOW INITIAL VALUES
windowWidth = 96
windowHeight = 160
windowShift = 10

scaleStart = 1
scaleEnd = 2
scaleStep = 0.5

#PREDICTION THRESHOLD
threshold = 0.5

#SLIDING WINDOW LOGIC
def getNextValidPosition(currentX, currentY, imgWidth, imgHeight, winWidth, winHeight, scale):
    currentX += windowShift
    if(currentX + winWidth >= imgWidth):
        currentX = 0
        currentY += windowShift
    if(currentY + winHeight >= imgHeight):
        currentX, currentY = 0, 0
        scale += scaleStep
        winWidth *= scale
        winHeight *= scale
    if(scale > scaleEnd): 
        currentX = -1 #END
    return currentX, currentY, winWidth, winHeight, scale

#CODE DOWNLOADED FROM http://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in xrange(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

            # delete all indexes from the index list that are in the
            # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return boxes[pick]

#THIS METHOD EXTRACTS SUBWINDOW, PERFORM PREDICTION AND LABELS DETECTED OBJECTS 
def extractPeopleInImage(imagePath, model):
    start_time = time.time()
    count_windows = 0
    
    #sliding window start position
    currentX, currentY = 0, 0
    winWidth = windowWidth
    winHeight = windowHeight
    scale = scaleStart

    img = Image.open(imagePath)
    imgWidth, imgHeight = img.size

    #creating clone image used for labeling detected objects
    clone = img.copy()
    draw = ImageDraw.Draw(clone)

    #array of detected objects, each detection is saved with 4 coordinates
    #top left and bottom right corner
    boxes = np.empty((0,4), int)

    while True:
        count_windows += 1
        #croping subwindow for prediction
        cropImg = img.crop((currentX,
                           currentY,
                           currentX + winWidth,
                           currentY + winHeight))
        #resizing subwindow if needed
        if(scale != 1):
            cropImg.thumbnail((windowWidth, windowHeight))

        #prediction
        pix = np.array(cropImg)
        pix = pix/255.
        output = model.predict(np.array([pix]))

        if output > 0.9:
            boxes = np.append(boxes, np.array([[currentX,
                                                currentY,
                                                currentX + winWidth,
                                                currentY + winHeight]]), axis = 0)

        #get next valid subwindow with sliding window technique
        currentX, currentY, winWidth, winHeight, scale = getNextValidPosition(currentX,
                                                                              currentY,
                                                                              imgWidth,
                                                                              imgHeight,
                                                                              winWidth,
                                                                              winHeight,
                                                                              scale)
        if(currentX < 0):
            break

    #removing redundant detected labels
    boxes = non_max_suppression_slow(boxes, threshold)

    #labeling image
    for box in boxes:
        draw.rectangle([box[0],
                        box[1],
                        box[2],
                        box[3]],
                        outline = (0, 255, 0))

    #saving predicted image
    clone.save(output_path)
    #clone.show()
    print 'TIME: ' + str(time.time() - start_time)
    print 'WINDOWS OPENED: ' + str(count_windows)

    
#MAIN PROGRAM : loads model and starts prediction
if __name__ == "__main__":
    model = load_model(modelPath)
    Tk().withdraw()
    imagePath = askopenfilename()
    extractPeopleInImage(imagePath, model)
