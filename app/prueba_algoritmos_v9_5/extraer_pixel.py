from app.declarar_librerias import *

def quitar_pixeles(image,tamanio_h,tamanio_w):
	promedio_w = 0
	promedio_h = 0
	cantidad = 0
	image_hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	umbral_bajo = (100,0,0)
	umbral_alto = (255,100,255)
	mask = cv2.inRange(image_hsv,umbral_bajo,umbral_alto)
	res = cv2.bitwise_and(image,image,mask=mask)
	ret, thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
	kernel = np.ones((5,5), np.uint8)
	thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel)
	_, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
	for i,cnt in enumerate(contours):
		x,y,w,h = cv2.boundingRect(cnt)
		if (w > tamanio_w and h > tamanio_h):
			cantidad = cantidad + 1
			promedio_w = promedio_w + w
			promedio_h = promedio_h + h
			area = cv2.contourArea(cnt)
			#print (w,h,area)
	if (cantidad != 0):
		promedio_w = promedio_w / cantidad
		promedio_h = promedio_h / cantidad
		
	for i,cnt in enumerate(contours):
		x,y,w,h = cv2.boundingRect(cnt)
		if (w >= promedio_w and h > promedio_h):
			area = cv2.contourArea(cnt)
	return w,h
