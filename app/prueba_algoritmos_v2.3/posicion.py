from declarar_librerias import *

def ubicacion(tam_x,tam_y,x,y,w,h):
	c_x = tam_x / 7
	c_y = tam_y / 7
	cuadrante_x = [c_x,c_x*2,c_x*3,c_x*4,c_x*5,c_x*6,c_x*7]
	cuadrante_y = [c_y,c_y*2,c_y*3,c_y*4,c_y*5,c_y*6,c_y*7]
	for i in range(0,7):
		for j in range(0,7):
			if (x < cuadrante_x[i] and y < cuadrante_y[j]):
				if (w < cuadrante_x[i] and h < cuadrante_y[j]):
					return i,j
				else:
					return i+1,j+1

def ubicacion2(tam_x,tam_y,x,y,w,h):
	mm_x = tam_x/8
	mm_y = tam_y/8
	pos_i = 0
	pos_j = 0
	x_cuadrante = 0
	y_cuadrante = 0
	cuadrante_x = [mm_x,mm_x*2,mm_x*3,mm_x*4,mm_x*5,mm_x*6,mm_x*7,mm_x*8,tam_x]
	cuadrante_y = [mm_y,mm_y*2,mm_y*3,mm_y*4,mm_y*5,mm_y*6,mm_y*7,mm_y*8,tam_y]
	for i in range(0,8):
		if (x <= cuadrante_x[i] and x+w <= cuadrante_x[i] and x_cuadrante == 0):
			pos_i = i
			x_cuadrante = 1
		elif (x <= cuadrante_x[i] and x+w <= cuadrante_x[i+1] and x_cuadrante == 0):
			dif_i = cuadrante_x[i] - x
			dif_i1 = x+w - cuadrante_x[i]
			if (dif_i >= dif_i1):
				pos_i = i
				x_cuadrante = 1
			else: 
				pos_i = i+1
				x_cuadrante = 1
	for j in range(0,8):
		if (y <= cuadrante_y[j] and y+h <= cuadrante_y[j] and y_cuadrante == 0):
			pos_j = j
			y_cuadrante = 1
		elif (y <= cuadrante_y[j] and y+h <= cuadrante_y[j+1]) and y_cuadrante == 0:
			dif_j = cuadrante_y[j] - y
			dif_j1 = y+h - cuadrante_y[j]
			if (dif_j >= dif_j1):
				pos_j = j
				y_cuadrante = 1
			else: 
				pos_j = j+1
				y_cuadrante = 1
	return pos_i,pos_j