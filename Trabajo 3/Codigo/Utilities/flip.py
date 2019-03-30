import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

mypath='Procesadas/YProcesadas'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread( join(mypath,onlyfiles[n]) )

    vertical_img = cv2.flip(images[n], 1)

    cv2.imwrite('Procesadas/YProcesadas2/'+str(n)+'.jpg',vertical_img)


    # wait time in milliseconds
    # this is required to show the image
    # 0 = wait indefinitely
    cv2.waitKey(1)

    # close the windows
    cv2.destroyAllWindows()