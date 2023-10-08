import cv2
import numpy as np
import matplotlib.pyplot as plt

# Normal routines
#imagen = cv2.imread('Fabi__24MP.jpg')
    
def square(img):
	#variables generales
	cX = 0
	cY = 0
	_x = 0
	_y = 0
	font = cv2.FONT_HERSHEY_SIMPLEX
	original = img.copy()
	original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#Establecemos el rango mínimo y máximo de H-S-V:
	amarillo_bajos = np.array([25,0,0])
	amarillo_altos = np.array([35,255,255])
	mask = cv2.inRange(hsv, amarillo_bajos, amarillo_altos)
	_, contours, hierachy = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
	#Encontrar el cuadrado 
	for i, cnt in enumerate (contours): 
		if hierachy[0,i,3] == -1:
			x,y,w,h = cv2.boundingRect(cnt)
			if (w > cX and h > cY):
				cX = w
				cY = h
				_x = x
				_y = y
				area = cv2.contourArea(cnt)
	#print (cX,cY)
	cv2.rectangle(mask, (_x,_y), (_x+cX,_y+cY), (0,255,0), 2)
	roi = original[_y:_y+cY,_x:_x+cX]
	# plt.imshow(roi)
	# plt.show()
	return roi,cX,cY,area

#img,w,h,_ = square(imagen)
#print ( w, h)
#print ("Eje X",cX)
#print ("Eje Y",cY)
#plt.imshow(img)
#plt.show()
