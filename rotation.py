import requests
import json
import io
import cv2
import imutils
from PIL import Image 

k=0
angle=0

API_KEY='API_KEY'

############################### tester les rotation avec angle 330 -> 360 avec un pas 5 degree########
for i in range (330,360,5):
	img = cv2.imread('exemple3.jpg')
	img=imutils.rotate(img,angle=i)
	cv2.imwrite('rotated.jpg', img)
	height, width, _ = img.shape
	roi = img[0: height, 0: width]
	url_api = "https://api.ocr.space/parse/image"
	_, compressedimage = cv2.imencode(".jpg", roi, [1, 90])
	file_bytes = io.BytesIO(compressedimage)
	result = requests.post(url_api,
              files = {'pic.jpg': file_bytes},
              data = {'apikey': API_KEY,
                      'language': "ara"})
	result = result.content.decode()
	result = json.loads(result)
	#print(result)
	if (len(result.get("ParsedResults"))!=0):
		parsed_results = result.get("ParsedResults")[0]
		text_detected = parsed_results.get("ParsedText")
		#word_list = text_detected.split()
		#number_of_words = len(word_list)		
		#print(number_of_words)
		
		print(len(text_detected))
		if (k<len(text_detected)):
			k=len(text_detected)
			angle=i
			with open ('space.txt','w',encoding='utf-8')as myfile:
				myfile.write(text_detected)	
############################### tester les rotations avec angle 0 -> 30 avec un pas 5 degree########	
for i in range (0,30,5):
	img = cv2.imread('exemple3.jpg')
	img=imutils.rotate(img,angle=i)
	cv2.imwrite('rotated_30.jpg', img)
	height, width, _ = img.shape
	roi = img[0: height, 0: width]
	url_api = "https://api.ocr.space/parse/image"
	_, compressedimage = cv2.imencode(".jpg", roi, [1, 90])
	file_bytes = io.BytesIO(compressedimage)
	result = requests.post(url_api,
              files = {'pic.jpg': file_bytes},
              data = {'apikey': API_KEY,
                      'language': "ara"})
	result = result.content.decode()
	result = json.loads(result)
	if (len(result.get("ParsedResults"))!=0):
		parsed_results = result.get("ParsedResults")[0]
		text_detected = parsed_results.get("ParsedText")
		#word_list = text_detected.split()
		#number_of_words = len(word_list)		
		#print(number_of_words)
		
		print(len(text_detected))
		if (k<len(text_detected)):
			k=len(text_detected)
			angle=i
			with open ('space.txt','w',encoding='utf-8')as myfile:
				myfile.write(text_detected)	
				
img = cv2.imread('exemple3.jpg')
img=imutils.rotate(img,angle=angle)
cv2.imwrite('final_pic.jpg', img)