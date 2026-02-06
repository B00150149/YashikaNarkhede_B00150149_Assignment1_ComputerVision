import cv2 as cv
import numpy as np
import time


#code the histogram, pick Threshold automatically
#manual version
def histogram(img):
    hist = np.zeros(256, dtype = int)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            hist[img[x,y]] += 1
    return hist



#automatic thresold (Question to ask:  Can use Otsu method Calculate between class variance)
#https://github.com/torlar/opencv-python-examples/blob/main/09_histograms/README.md
def threshold_auto(hist):
    #find peak values
    background_peak = np.argmax(hist[:128])
    foreground_peak = np.argmax(hist[128:]) + 128
    #Calculate the threshold as midpoint
    threshold = (background_peak + foreground_peak) // 2
    return threshold
    # return int ((background_peak + foreground_peak) / 2)



#Threshholding
def threshold (img, thresh):
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x,y] > thresh:
                img[x,y] = 255
            else:
                img[x,y] = 0
    return img



#Main 
for i in range(1,16):
    #read in an image into memory
    img = cv.imread('C:/Users/yashi/Documents/01_tud_yashika/4th_year/Computer Vision/lab1/Oring' + str(i) + '.jpg',0)
    
    hist = histogram(img)
    thresh = threshold_auto(hist)
    bw = threshold(img, thresh)
   
    rgb = cv.cvtColor(bw,cv.COLOR_GRAY2RGB)
    cv.putText(rgb, "Image:" + str(i) , (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv.circle(rgb, (40, 40), 20, (0,0,255))
    cv.imshow('thresholded image', rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()
