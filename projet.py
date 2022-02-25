import os
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
from PIL import Image, ImageStat
import time


def brightness_calculator( im_file ):
   im = Image.open(im_file).convert('L')
   stat = ImageStat.Stat(im)
   return stat.mean[0]

   
def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))

################################### eliminer le flash####################
img = cv2.imread('exemple5.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayimg = gray
#GLARE_MIN = np.array([0, 0, 50],np.uint8)
#GLARE_MAX = np.array([0, 0, 225],np.uint8)
#hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)
mask1 = cv2.threshold(grayimg , 220, 255, cv2.THRESH_BINARY)[1]
result1 = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA) 
	#cv2.imshow("Image without glare", result1)
cv2.imwrite('exemple5_without_glare.jpg', result1)
	
###########################  ajouter brightness ###################	
#########################   (BACKGROUND MUST BE OFF) #################
brightness_=brightness_calculator('exemple5_without_glare.jpg')
print(brightness_)
if (brightness_>200):
		value=0
elif(brightness_>170):
		value=10
elif(brightness_>150):
		value=17
elif(brightness_>140):
		value=24
elif(brightness_>130):
		value=31
elif(brightness_>120):
		value=38
elif(brightness_>110):
		value=46	
elif(brightness_>100):
		value=53
elif(brightness_>90):
		value=60
elif(brightness_>80):
		value=67
elif(brightness_>70):
		value=74
elif(brightness_>60):
		value=81
elif(brightness_>50):
		value=88
elif(brightness_>40):
		value=95
elif(brightness_>30):
		value=102
elif(brightness_>20):
		value=109
elif(brightness_>10):
		value=116		
else:
		value=123
image = increase_brightness(result1, value)
cv2.imshow('brightness',image)
####################### modifier l'image en eliminant les champs nom prenom date naissance....######	

lower_black = np.array([0,0,0], dtype = "uint16")
upper_black = np.array([110,110,110], dtype = "uint16")
modified_img =255- cv2.inRange(image, lower_black, upper_black)
cv2.imshow('image_without_labels',modified_img)
cv2.imwrite('test_exemple5.jpg', filtred_img)
	
#reader = easyocr.Reader(['ar'])
#result = reader.readtext('MyImage.jpg',paragraph="False")
#with open ('file4.txt','w',encoding='utf-8')as myfile:
#	myfile.write(str(result))
cv2.waitKey(0)
cv2.destroyAllWindows()
	
#################################### ELiminer le champ nom,prenom, date naissance.... de la cin######

"""
lower_range_black = np.array([0,0,0], dtype = "uint16")
upper_range_black = np.array([110,110,110], dtype = "uint16")
filtred_img =255- cv2.inRange(image, lower_range_black, upper_range_black)
cv2.imshow('Pic_BW_without_labels',filtred_img)
cv2.imwrite('test5.jpg', filtred_img)
"""
	
	


################################# affichage des contours par ordre decroissant ################
""" 
img = cv2.imread('exemple4.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bilateral = cv2.bilateralFilter(gray, 5, 5,5)
eq = cv2.equalizeHist(bilateral)
edged = cv2.Canny(eq, 0, 150)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
h, w = img.shape[:2]
thresh_area = 0.001
list_contours = list()
for c in contours:
    area = cv2.contourArea(c)

    if (area > thresh_area*h*w): 
        rect_page = cv2.minAreaRect(c)
        box_page = np.int0(cv2.boxPoints(rect_page))
        list_contours.append(box_page)
		
sorted_contours= sorted(list_contours, key=cv2.contourArea, reverse= True)
#
for (i,c) in enumerate(sorted_contours):
    x,y,w,h= cv2.boundingRect(c)
    
    cropped_contour= img[y:y+h, x:x+w]
    cv2.imwrite("contour.jpg", cropped_contour)
    readimage= cv2.imread("contour.jpg")
    cv2.imshow('Image', readimage)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
"""

################################ IMAGE BLACK AND WHITE ##################
#image = cv2.imread(IMAGE_PTH)
#img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY)
#cv2.imshow('image BW', thresh)
#cv2.waitKey(0)
#cv2.imwrite('image_BW.jpg', thresh)
#cv2.destroyAllWindows()