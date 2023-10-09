from prueba_algoritmos_v9_5.declarar_librerias import * 

def calibre(img):
	w_= 0
	h_= 0
	radius_2 = 0
	b = 0
	angle = 0
	angulo = 0
	# Mascara para ignorar el fondo blanco
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	mask = mask // 255

	#eliminar el fondo blanco
	b,g,r = cv2.split(img)
	b = b * mask
	g = g *mask 
	r = r * mask
	relevant = cv2.merge((b,g,r)).astype(np.uint8)

	#separar los canales
	b,g,r = cv2.split(relevant)
	mix = 0.9*r+0.1*g
	mix = mix.astype(np.uint8)

	#contorno
	ret, thresh = cv2.threshold(mix,1,255,cv2.THRESH_BINARY)
	kernel = np.ones((5,5), np.uint8)
	thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel)
	
	#src = cv2.medianBlur(img, 5)
	src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	_, contours, hierachy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	for index, cnt in enumerate(contours):
		x,y,w,h = cv2.boundingRect(cnt)
		if (len(cnt) >= 5):
			(xc,xy),(a,b),angulo= cv2.fitEllipse(cnt)
			#elipse = cv2.fitEllipse(cnt)
			#cv2.ellipse(img,elipse,(0,255,0),2)
			if (b >= radius_2):
				radius_2 = b
			if (angulo >= angle):
				angle = angulo

		if (w > w_):
			w_ = w
		if (h > h_):
			h_ = h

	return w_,h_,radius_2,angle
