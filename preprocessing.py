import os, time
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from typing import List, Text

IMAGE_DIR = os.path.join('..', 'data', 'breast', 'Train')
IMAGE_PROCESSED = os.path.join('..', 'data', 'breast_processed')

def list_pictures(directory: Text, mag, ext='.png') -> List:
    # return a list contains paths of all 400*400 histopathologic image 
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if f.endswith(ext)
            if f.split('-')[3] == mag]

def gaussian_blur(image):
    return cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)

def rgb_to_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    #lower_blue = np.array([110,50,50])
    #upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    #mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(image, image, mask= mask)
    return hsv


def preprocessing(image_dirs, target_dir, zoom):
    for image_dir in tqdm(list_pictures(image_dirs, zoom)):
        image = cv2.imread(image_dir)
        image = gaussian_blur(image)
        # transfer the image from rgb to gray for histogram equlization
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #equ = cv2.equalizeHist(gray)
        # transfer the image from rgb to hsv
        #hsv = rgb_to_hsv(image)
        # save the pre-processed image 
        directory, file_name = os.path.split(image_dir)
        #sub_directory = directory.split(os.sep)[4]
        cv2.imwrite(os.path.join(target_dir, file_name), image)
        #cv2.imwrite(os.path.join(target_dir, 'hsv_' + file_name), hsv)
        #cv2.imwrite(os.path.join(target_dir, 'equ_' + file_name), equ)
        #k = cv2.waitKey(1005) & 0xFF
        #if k == 27:
        #    break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for factor in ['40', '100', '200', '400']:
        multi1 = time.time()
        print('Parent process {0}'.format(os.getpid()))
        p = Pool(2)   # create two sub processes
        p.apply_async(preprocessing,
                      args=(os.path.join(IMAGE_DIR),
                            os.path.join(IMAGE_PROCESSED, factor),
                            factor))
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')
        multi2 = time.time()
        print('Time comsumed:', multi2 - multi1)


