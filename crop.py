import cv2
import numpy as np


############################################## LITTLE FLAG + Carte grise##########################
gr=cv2.imread('grise12.png')
hsv = cv2.cvtColor(gr, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 10, 120])
upper_red = np.array([15, 255, 255])
mask = cv2.inRange (hsv, lower_red, upper_red)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
h, w = gr.shape[:2]
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
    if(w>h)&(h*2>w)&(h<800)&(w<800):
	    cropped_contour= gr[y:y+h, x:x+w]
		# pour savoir la couleur domiante
	    pixels = np.float32(cropped_contour.reshape(-1, 3))
	    n_colors = 7
	    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	    flags = cv2.KMEANS_RANDOM_CENTERS
	    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
	    _, counts = np.unique(labels, return_counts=True)
	    dominant = palette[np.argmax(counts)]
	    print(dominant)
	    # si le rouge est plus dominant que le vert
	    if(dominant[1]*1.8<dominant[2]):
			#Si le drapeau existe l'image finale est correcte  
		    #cv2.imshow('Flag', cropped_contour)
		    #cv2.imwrite("flag_contour5.jpg", cropped_contour)
		    Top=round(y+h*1.1)
		    Bottom=round(y+h*7)
		    Left=round(x-w*3.8)
		    Right=round(x+w*4)
		    final_pic=gr[Top:Bottom,Left:Right]
		    cv2.imshow('Image_finale', final_pic)
		    cv2.imwrite("Zoomed_carte_grise.jpg", final_pic)
		    cv2.waitKey(0)
    
cv2.destroyAllWindows()

##########################################################################################
