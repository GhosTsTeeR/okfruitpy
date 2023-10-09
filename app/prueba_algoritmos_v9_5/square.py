from prueba_algoritmos_v9_5.declarar_librerias import *

def square(img):
	#variables generales
	cX = 0
	cY = 0
	_x = 0
	_y = 0
	font = cv2.FONT_HERSHEY_SIMPLEX
	
	#Cambiamos los canales de la imagen
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# ## Establecemos el rango mínimo y máximo de H-S-V:
	# verdes_bajos = np.array([40,0,0])
	# verdes_altos = np.array([80,255,255])
	# mask = cv2.inRange(hsv, verdes_bajos, verdes_altos)
	
	#Recordatorio: El rango HSV funciona de la siguiente forma:
	#-La 1a componente es la tonalidad (Hue), en nuestro caso amarillo.
	#-La 2a componente es la saturación (Saturation) , que hace el color + o - grisáceo.
	#-La 3a componente es el valor (Value), que hace el color + o - oscuro.
	 
	## Detectamos los píxeles que estén dentro del rango que hemos establecido:
	azules_bajos = np.array([110,50,50])
	azules_altos = np.array([130,255,255])
	mask = cv2.inRange(hsv, azules_bajos, azules_altos)

	_, contours, hierachy = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

	#Encontrar el cuadrado 
	for i, cnt in enumerate (contours): 
		if hierachy[0,i,3] == -1:
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 8)
			if (w > cX and h > cY):
				cX = w
				cY = h
				_x = x
				_y = y

	#print (cX,cY)
	#cv2.rectangle(mask, (_x,_y), (_x+cX,_y+cY), (0,255,0), 2)
	#plt.imshow(mask)
	#plt.show()
	return img,cX,cY


#imagen,X,Y = square(img)
#plt.imshow(imagen)
#plt.show()
#plt.imshow(mascara1)
#plt.show()

