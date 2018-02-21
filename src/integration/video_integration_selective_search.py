import numpy as np
from keras.models import load_model
from Tkinter import Tk
from tkFileDialog import askopenfilename
from PIL import Image, ImageDraw
import os
import time
import imageio
import selectivesearch

#GLOBAL VARIABLES
modelPath = "../../models/slozenija_20_vise_slika.h5"

video_source_path = "../../VIDEO/m/MP4/cut.mp4"
img_clipped_path = "../../VIDEO/m/IMG/"
img_labeled_path = "../../VIDEO/m/IMG_L/"
video_dest_path = "../../VIDEO/m/MP4_L/a.gif"
framesPerSec = 25


#SLIDING WINDOW INITIAL VALUES
windowWidth = 96
windowHeight = 160
windowShift = 10

#PREDICTION THRESHOLD
threshold = 0.5

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

# RETURNS IMAGE WITH WHITE BORDER AND OFFSET TO OLD PIC COORDINATES
def make_border(img):
    old_x, old_y = img.size
    new_x = int(old_x * 1.4)
    new_y = int(old_y * 1.4)
    
    new_img = Image.new("RGB", (new_x, new_y), color = (255, 255, 255))
    new_img.paste(img, ((new_x-old_x)/2, (new_y-old_y)/2))

    diff_x = int((new_x - old_x) / 2)
    diff_y = int((new_y - old_y) / 2)

    return new_img, diff_x, diff_y

def crop_img_with_border(img, x, y, w, h):
    #   MAKE BORDER TO ORIGINAL IMAGE
    img, offset_x, offset_y = make_border(img)

    x = x + offset_x
    y = y + offset_y
    
    wanted_width = int(w * 0.35)
    wanted_height = int(h * 0.35)

    cropImg = img.crop((x - wanted_width,
                        y - wanted_height,
                        x + w + wanted_width,
                        y + h + wanted_height))
    return cropImg   

#THIS METHOD EXTRACTS SUBWINDOW, PERFORM PREDICTION AND LABELS DETECTED OBJECTS 
def extractPeopleInImage(imagePath, model):
    img = Image.open(img_clipped_path + imagePath)
    imgWidth, imgHeight = img.size

    #creating clone image used for labeling detected objects
    clone = img.copy()
    draw = ImageDraw.Draw(clone)

    #array of detected objects, each detection is saved with 4 coordinates
    #top left and bottom right corner
    boxes = np.empty((0,4), int)

    #EXTRACT WINDOWS TO PREDICT USING SELECTIVE SEARCH
    img_np = np.array(img)
    img_lbl, regions = selectivesearch.selective_search(img_np, scale=100, sigma=0.8, min_size=50)

    candidates = set()
    #FILTERING CANDIDATES
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 0.8 or h / w > 2:
            continue
        candidates.add(r['rect'])

    #PERFORMING PREDICTION FOR ALL CANDIDATES
    for c in candidates:
        x, y, w, h = c

        cropImg = crop_img_with_border(img, x, y, w, h)
        
        #resizing subwindow if needed
        cropImg = cropImg.resize((windowWidth, windowHeight))

        #prediction
        pix = np.array(cropImg)
        pix = pix/255.
        output = model.predict(np.array([pix]))

            
        if output > 0.9:
            boxes = np.append(boxes, np.array([[x,
                                                y,
                                                x + w,
                                                y + h]]),
                                                axis = 0)    

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
    clone.save(img_labeled_path + imagePath, 'JPEG')

def video_to_img():
    vid = imageio.get_reader(video_source_path, 'ffmpeg')

    for i, im in enumerate(vid):
        image = Image.fromarray(im, 'RGB')
        dest = img_clipped_path + str(i) + ".jpg"
        image.save(dest, 'JPEG')

def img_to_video():
    img_array = []

    img_index = [int(x.split(".")[0]) for x in os.listdir(img_labeled_path)] 
    img_index = sorted(img_index)
    img_path = [str(x) + ".jpg" for x in img_index]

    for path in img_path:
    
        img = imageio.imread(img_labeled_path + path, 'JPEG')
        img_array.append(img)


    imageio.mimsave(video_dest_path, img_array, 'GIF', fps=framesPerSec)

if __name__ == "__main__":
    video_to_img()
    
    startTime = time.time()
    model = load_model(modelPath)
    count = 1
    l = len(os.listdir(img_clipped_path))
    
    #generating predictions for all frames from video
    for img in os.listdir(img_clipped_path):
        print 'Img ' + str(count) + '/' + str(l)
        extractPeopleInImage(img, model)
        count += 1
    print ("--- %s seconds ---" % (time.time() - startTime))

    img_to_video()
