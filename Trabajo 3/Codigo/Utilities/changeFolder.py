import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

mypath='DB/zz'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
h = 232
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread( join(mypath,onlyfiles[n]) )

    #vertical_img = cv2.flip(images[n], 1)

    cv2.imwrite('DB/Y/'+str(n+h)+'.jpg', images[n])
