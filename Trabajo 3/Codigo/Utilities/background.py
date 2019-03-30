import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

mypath='ImagesToTrain/Y'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    original = cv2.imread( join(mypath,onlyfiles[n]) )
    images[n] = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 33, 20], dtype=np.uint8)
    upper_white = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(images[n], lower_white, upper_white)  # could also use threshold
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))  # "erase" the small white points in the resulting mask
#    mask = cv2.bitwise_not(mask)  # invert mask

    mask = cv2.GaussianBlur(mask, (3,3), 0)
    skin = cv2.bitwise_and(original, original, mask = mask)

    bk = np.full(skin.shape, 255, dtype=np.uint8)  # white bk

    # get masked foreground
    fg_masked = cv2.bitwise_and(original, original, mask=mask)

    # get masked background, mask must be inverted
 #   mask = cv2.bitwise_not(mask)
    bk_masked = cv2.bitwise_and(bk, bk, mask=mask)

    # combine masked foreground and masked background
#    final = cv2.bitwise_or(skin, bk_masked)
#    mask = cv2.bitwise_not(mask)  # revert mask to original

    #cv2.imshow('alv3', cv2.resize(final, (300, 300)))
    skin[skin == 0] = 255
    cv2.imwrite('Procesadas/YProcesadas/'+str(n)+'.jpg', skin)

    cv2.waitKey(1)