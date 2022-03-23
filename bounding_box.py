import cv2 as cv
import numpy as np


src = cv.imread(cv.samples.findFile('CBIR/CBIR/image_fashion/24730.jpg'))
if src is None:
    print('Could not open or find the image:', 'CBIR/CBIR/image_fashion/24730.jpg')
    exit(0)
#downscale the images
print(src.shape)
width = int(src.shape[1] / 3)
height = int(src.shape[0] / 3)
scr_rescaled = cv.resize(src, (width, height))
# Convert image to gray and blur it
src_gray = cv.cvtColor(scr_rescaled, cv.COLOR_BGR2GRAY)
src_gray = cv.GaussianBlur(src_gray, (3,3),0)
""" cv.imshow('raw',scr_rescaled)
cv.waitKey(0)
cv.imshow('gray scale',src_gray)
cv.waitKey(0) """

def funcCan(thresh1=0):
    thresh1 = cv.getTrackbarPos('thresh1', 'canny')
    thresh2 = cv.getTrackbarPos('thresh2', 'canny')
    img_canny = cv.Canny(src_gray, thresh1, thresh2)
    cv.imshow('canny', img_canny)
def get_countours(img, ouputimage):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 2000:
            cv.drawContours(ouputimage, contours, 0, (255,0,255),5)
            pr = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.005 * pr, True)
            x, y, w, h = cv.boundingRect(approx)
            cv.rectangle(ouputimage, (x,y),(x + w, y + h), (0,255,0), 5)

thresh1=52
thresh2=26
cv.namedWindow('canny')
cv.createTrackbar('thresh1','canny',thresh1,255,funcCan)
cv.createTrackbar('thresh2','canny',thresh2,255,funcCan)
funcCan(0)
img_canny = cv.Canny(src_gray, thresh1, thresh2)
kernel = np.ones((5,5))
img_dill = cv.dilate(img_canny,kernel, iterations=1)
cv.imshow('Frame',img_dill)
cv.waitKey(0)
ouputimage = scr_rescaled.copy()
get_countours(img_dill, ouputimage)
cv.imshow('Frame2',ouputimage)
cv.waitKey(0)
print('END')