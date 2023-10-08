import numpy as np
import cv2
from app.prueba_algoritmos_v2_3.color import *
import argparse
from reportlab.pdfgen import canvas
from square_3 import square
import pandas as pd
#import zbar
#scanner = zbar.Scanner()

import itertools
from reportlab.lib.pagesizes import A4


#Variables globales
font = cv2.FONT_HERSHEY_SIMPLEX
numero_fruto = 1
referencia_imagen = 0
cantidad_frutos_max = 100
#Vraibles globales para evitar el error de segmentación
tamanio_w = 0
tamanio_h = 0
cantidad = 0
area_prom = 0
prom_w = 0
prom_h = 0
cantidad_ima = 0
roi_list  = []
hist_list = []
results = []

#leer la imagen
parser = argparse.ArgumentParser(description='Image read.')
parser.add_argument('--image', '-i',help='Image to open.')
args = parser.parse_args()
name = args.image
image_file = args.image
img = cv2.imread(image_file)
tam_y,tam_x,_ = img.shape

#se hacen copias de las imagenes para los reportes
original = img.copy()
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
original_hsv = cv2.cvtColor(original,cv2.COLOR_RGB2HSV)
muestra = img.copy()


## EVALUAR OPTIMIZAR RESULTADOS MEJORANDO ESTE FILTRO.
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
ret2, th = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
mask1 = cv2.bitwise_and(img, img, mask = th)
white = np.zeros_like(img)
white = cv2.bitwise_not(white)
mask2 = cv2.bitwise_and(white, white, mask = cv2.bitwise_not(th))
img = mask2 + mask1

#eliminar el amarillo
#mask_amarilla = cv2.inRange(original_hsv,(15,0,0),(36,255,255)) #este filtro se debe calibrar con la caja Amarillo
#target = cv2.bitwise_not(img,img,mask=mask_amarilla)
#plt.imshow(target)
#plt.show()

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
#plt.imshow(thresh)
#plt.show()

_, contours, hierachy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()

#Verificación para eliminar ruido de la imagen (Contornos pequeños)
for index, cnt in enumerate(contours):
  x,y,w,h = cv2.boundingRect(cnt)
  tamanio_w = tamanio_w + w
  tamanio_h = tamanio_h + h
  cantidad = cantidad + 1

tamanio_w = tamanio_w / cantidad
tamanio_h = tamanio_h / cantidad
cantidad = 0

#Verificación para eliminar ruido de la imagen (Contornos pequeños)
for index, cnt in enumerate(contours):
  x,y,w,h = cv2.boundingRect(cnt)
  if (w > tamanio_w and h > tamanio_h):
      cantidad = cantidad + 1
      area = cv2.contourArea(cnt)
      area_prom = area_prom + area

area_prom = area_prom / cantidad
cantidad = 0

print (tamanio_w,tamanio_h)
for index, cnt in enumerate(contours):
  x,y,w,h = cv2.boundingRect(cnt)
  cv2.rectangle(output,(x,y),(x+w,y+h),(255,255,0),2)
  if (w > tamanio_w and h > tamanio_h and (w*h) > (tamanio_h*tamanio_w)):
    #Se define el radio de las circunferencias a analizar 
    (xc, yc), radius = cv2.minEnclosingCircle(cnt)
    area = cv2.contourArea(cnt)
    if ((w*h)  > area_prom/2):
        if (w > h):
            diferencia = w - h
            valor = h
        if (h >= w):
            diferencia = h - w
            valor = w
        if (diferencia < valor):
            cantidad = cantidad + 1
            prom_w = prom_w + w
            prom_h = prom_h + h 
            

prom_w = (prom_w / cantidad)
prom_h = (prom_h / cantidad)

#escalamos el tamanio de los pixeles
img_square, cX, cY, area_square = square(muestra)
#el cuadrado mide 5cm se escala para saber el tamaño de los pixeles
cX =  (float(5/cX))
cY =  (float(5/cY))
c_area = (float(16/area_square))
#Verificación de los contornos para encontrar los frutos
contador_fruto = 0
for index, cnt in enumerate(contours):
  x,y,w,h = cv2.boundingRect(cnt)
  cv2.rectangle(output,(x,y),(x+w,y+h),(255,255,0),2)
  if (w > tamanio_w and h > tamanio_h and (w*h) > (tamanio_h*tamanio_w)):
    #Se define el radio de las circunferencias a analizar 
    (xc, yc), radius = cv2.minEnclosingCircle(cnt)
    area = cv2.contourArea(cnt)
    if ((w*h)  > area_prom/2):
        if (w > h):
            diferencia = w - h
            valor = h
        if (h >= w):
            diferencia = h - w
            valor = w
        if (diferencia < valor):
           if (w < prom_w*1.5 and h < prom_h*1.5):
                  contador_fruto = contador_fruto + 1
                  roi = original[y:y+h, x:x+w]
                  roi_mask = thresh[y:y+h, x:x+w] // 255
                  roi_list.append(roi)
                  color_class = getClass(roi,roi_mask)
                  results.append({
                     'x'           : x,
                     'y'           : y,
                     'w'           : w,
                     'h'           : h,
                     'size_x'      : w*cX,
                     'size_y'      : h*cY,
                     'color_class' : color_class,
                     'radius'      : radius })

data_frame = pd.DataFrame(results)
data_2 = data_frame.reset_index(drop = True).sort_values(by = 'y', ascending=False)
y_max = data_2['y'].max()
h_min = data_2['h'].min()
lista_contadores = []
h_max = data_2['h'].max()
print (h_min,h_max)
contador_c1 = 0
total_segmentaciones = len(data_2)
for i in data_2.index:
    if ((y_max - data_2['y'][i]) < h_min):
       #print (data_2['y'][i])
       contador_c1 = contador_c1 + 1
    else:
       y_max = data_2['y'][i]
       lista_contadores.append(contador_c1)
       contador_c1 = contador_c1 + 1
lista_contadores.append(total_segmentaciones)
print (lista_contadores)
	
if (len(lista_contadores) == 7):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	d7 = data_frame[lista_contadores[5]:lista_contadores[6]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6,d7])

if (len(lista_contadores) == 8):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	d7 = data_frame[lista_contadores[5]:lista_contadores[6]].sort_values('x')
	d8 = data_frame[lista_contadores[6]:lista_contadores[7]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8])

if (len(lista_contadores) == 9):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	d7 = data_frame[lista_contadores[5]:lista_contadores[6]].sort_values('x')
	d8 = data_frame[lista_contadores[6]:lista_contadores[7]].sort_values('x')
	d9 = data_frame[lista_contadores[7]:lista_contadores[8]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9])
	
if (len(lista_contadores) == 10):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	d7 = data_frame[lista_contadores[5]:lista_contadores[6]].sort_values('x')
	d8 = data_frame[lista_contadores[6]:lista_contadores[7]].sort_values('x')
	d9 = data_frame[lista_contadores[7]:lista_contadores[8]].sort_values('x')
	d10 = data_frame[lista_contadores[8]:lista_contadores[9]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10])
	
if (len(lista_contadores) == 11):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	d7 = data_frame[lista_contadores[5]:lista_contadores[6]].sort_values('x')
	d8 = data_frame[lista_contadores[6]:lista_contadores[7]].sort_values('x')
	d9 = data_frame[lista_contadores[7]:lista_contadores[8]].sort_values('x')
	d10 = data_frame[lista_contadores[8]:lista_contadores[9]].sort_values('x')
	d11 = data_frame[lista_contadores[9]:lista_contadores[10]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11])

if (len(lista_contadores) == 12):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	d7 = data_frame[lista_contadores[5]:lista_contadores[6]].sort_values('x')
	d8 = data_frame[lista_contadores[6]:lista_contadores[7]].sort_values('x')
	d9 = data_frame[lista_contadores[7]:lista_contadores[8]].sort_values('x')
	d10 = data_frame[lista_contadores[8]:lista_contadores[9]].sort_values('x')
	d11 = data_frame[lista_contadores[9]:lista_contadores[10]].sort_values('x')
	d12 = data_frame[lista_contadores[10]:lista_contadores[11]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12])

if (len(lista_contadores) == 13):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	d7 = data_frame[lista_contadores[5]:lista_contadores[6]].sort_values('x')
	d8 = data_frame[lista_contadores[6]:lista_contadores[7]].sort_values('x')
	d9 = data_frame[lista_contadores[7]:lista_contadores[8]].sort_values('x')
	d10 = data_frame[lista_contadores[8]:lista_contadores[9]].sort_values('x')
	d11 = data_frame[lista_contadores[9]:lista_contadores[10]].sort_values('x')
	d12 = data_frame[lista_contadores[10]:lista_contadores[11]].sort_values('x')
	d13 = data_frame[lista_contadores[11]:lista_contadores[12]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13])
	
if (len(lista_contadores) == 14):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	d7 = data_frame[lista_contadores[5]:lista_contadores[6]].sort_values('x')
	d8 = data_frame[lista_contadores[6]:lista_contadores[7]].sort_values('x')
	d9 = data_frame[lista_contadores[7]:lista_contadores[8]].sort_values('x')
	d10 = data_frame[lista_contadores[8]:lista_contadores[9]].sort_values('x')
	d11 = data_frame[lista_contadores[9]:lista_contadores[10]].sort_values('x')
	d12 = data_frame[lista_contadores[10]:lista_contadores[11]].sort_values('x')
	d13 = data_frame[lista_contadores[11]:lista_contadores[12]].sort_values('x')
	d14 = data_frame[lista_contadores[12]:lista_contadores[13]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14])
	
if (len(lista_contadores) == 15):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	d7 = data_frame[lista_contadores[5]:lista_contadores[6]].sort_values('x')
	d8 = data_frame[lista_contadores[6]:lista_contadores[7]].sort_values('x')
	d9 = data_frame[lista_contadores[7]:lista_contadores[8]].sort_values('x')
	d10 = data_frame[lista_contadores[8]:lista_contadores[9]].sort_values('x')
	d11 = data_frame[lista_contadores[9]:lista_contadores[10]].sort_values('x')
	d12 = data_frame[lista_contadores[10]:lista_contadores[11]].sort_values('x')
	d13 = data_frame[lista_contadores[11]:lista_contadores[12]].sort_values('x')
	d14 = data_frame[lista_contadores[12]:lista_contadores[13]].sort_values('x')
	d15 = data_frame[lista_contadores[13]:lista_contadores[14]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15])
	
if (len(lista_contadores) == 16):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	d7 = data_frame[lista_contadores[5]:lista_contadores[6]].sort_values('x')
	d8 = data_frame[lista_contadores[6]:lista_contadores[7]].sort_values('x')
	d9 = data_frame[lista_contadores[7]:lista_contadores[8]].sort_values('x')
	d10 = data_frame[lista_contadores[8]:lista_contadores[9]].sort_values('x')
	d11 = data_frame[lista_contadores[9]:lista_contadores[10]].sort_values('x')
	d12 = data_frame[lista_contadores[10]:lista_contadores[11]].sort_values('x')
	d13 = data_frame[lista_contadores[11]:lista_contadores[12]].sort_values('x')
	d14 = data_frame[lista_contadores[12]:lista_contadores[13]].sort_values('x')
	d15 = data_frame[lista_contadores[13]:lista_contadores[14]].sort_values('x')
	d16 = data_frame[lista_contadores[14]:lista_contadores[15]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16])
	
if (len(lista_contadores) == 17):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	d7 = data_frame[lista_contadores[5]:lista_contadores[6]].sort_values('x')
	d8 = data_frame[lista_contadores[6]:lista_contadores[7]].sort_values('x')
	d9 = data_frame[lista_contadores[7]:lista_contadores[8]].sort_values('x')
	d10 = data_frame[lista_contadores[8]:lista_contadores[9]].sort_values('x')
	d11 = data_frame[lista_contadores[9]:lista_contadores[10]].sort_values('x')
	d12 = data_frame[lista_contadores[10]:lista_contadores[11]].sort_values('x')
	d13 = data_frame[lista_contadores[11]:lista_contadores[12]].sort_values('x')
	d14 = data_frame[lista_contadores[12]:lista_contadores[13]].sort_values('x')
	d15 = data_frame[lista_contadores[13]:lista_contadores[14]].sort_values('x')
	d16 = data_frame[lista_contadores[14]:lista_contadores[15]].sort_values('x')
	d17 = data_frame[lista_contadores[15]:lista_contadores[16]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17])
	
	
if (len(lista_contadores) >= 18):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	d7 = data_frame[lista_contadores[5]:lista_contadores[6]].sort_values('x')
	d8 = data_frame[lista_contadores[6]:lista_contadores[7]].sort_values('x')
	d9 = data_frame[lista_contadores[7]:lista_contadores[8]].sort_values('x')
	d10 = data_frame[lista_contadores[8]:lista_contadores[9]].sort_values('x')
	d11 = data_frame[lista_contadores[9]:lista_contadores[10]].sort_values('x')
	d12 = data_frame[lista_contadores[10]:lista_contadores[11]].sort_values('x')
	d13 = data_frame[lista_contadores[11]:lista_contadores[12]].sort_values('x')
	d14 = data_frame[lista_contadores[12]:lista_contadores[13]].sort_values('x')
	d15 = data_frame[lista_contadores[13]:lista_contadores[14]].sort_values('x')
	d16 = data_frame[lista_contadores[14]:lista_contadores[15]].sort_values('x')
	d17 = data_frame[lista_contadores[15]:lista_contadores[16]].sort_values('x')
	d18 = data_frame[lista_contadores[16]:lista_contadores[17]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18])

frame_cereza2 = frame_cereza.to_records()


for r in frame_cereza2:
  x = r['x']
  y = r['y']
  w = r['w']
  h = r['h']
  
  cv2.rectangle(original, (x,y), (x+w,y+h), (255,255,0), 2)
  cv2.putText(original, str(numero_fruto), (x,y), font, 1.8, (255,0,0), 3, cv2.LINE_AA) #Aqui puedes cambiar el tamaño de los números de 0.7 a 1.5 recomendado y el grosor de 2 a 3
  numero_fruto = numero_fruto + 1

img_result = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
cv2.imwrite("Resultado_"+image_file, img_result)

def grouper(iterable,n):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args)

def export_to_pdf(data,image_file,lista_color,lista_calibre):
    c = canvas.Canvas("Reporte_Arandano_"+image_file+".pdf", pagesize=A4)
    c.drawString(400,800, "Reporte Arandano")
    c.drawString(120,750, "Resumen de color")
    for i in range(0,4):
        if (lista_color[0] >= 0):
           c.drawString(100,700, "Porcentaje de Arándanos de Color Verde : "+ str(lista_color[0])+" %")
        if (lista_color[1] >= 0):
           c.drawString(100,680, "Porcentaje de Arándanos de Color Roja Punta Amarillo : "+ str(lista_color[1])+" %")
        if (lista_color[2] >= 0):
           c.drawString(100,660, "Porcentaje de Arándanos de Color Rojo : "+ str(lista_color[2])+" %")
        if (lista_color[3] >= 0):
           c.drawString(100,640, "Porcentaje de Arándanos de Color Purpura : "+ str(lista_color[3])+" %")
        if (lista_color[4] >= 0):
           c.drawString(100,620, "Porcentaje de Arándanos de Color Optimo Exportación : "+ str(lista_color[4])+" %")
    c.drawString(120,580, "Resumen de Calibre (mm) ")
    for i in range(0,4):
        if (lista_calibre[0] >= 0):
           c.drawString(100,530, "Porcentaje de Arándanos con tamaño inferior a 12 mm : "+ str(lista_calibre[0])+" %")
        if (lista_calibre[1] >= 0):
           c.drawString(100,510, "Porcentaje de Arándanos con tamaño entre 12 y 14 mm : "+ str(lista_calibre[1])+" %")
        if (lista_calibre[2] >= 0):
           c.drawString(100,490, "Porcentaje de Arándanos con tamaño entre 14 y 16 mm : "+ str(lista_calibre[2])+" %")
        if (lista_calibre[3] >= 0):
           c.drawString(100,470, "Porcentaje de Arándanos con tamaño entre 16 y 18 mm : "+ str(lista_calibre[3])+" %")
        if (lista_calibre[4] >= 0):
           c.drawString(100,450, "Porcentaje de Arándanos con tamaño entre 18 y 20 mm : "+ str(lista_calibre[4])+" %")
        if (lista_calibre[5] >= 0):
           c.drawString(100,430, "Porcentaje de Arándanos con tamaño entre 20 y 22 mm : "+ str(lista_calibre[5])+" %")
        if (lista_calibre[6] >= 0):
           c.drawString(100,410, "Porcentaje de Arándanos con tamaño superior a 22 mm : "+ str(lista_calibre[6])+" %")   
    c.showPage() 
    a, b = A4
    max_rows_per_page = 45
    x_offset = 50
    y_offset = 80
    padding = 15
    x_list = [ x + x_offset for x in [0,150,300,450]]
    y_list = [ b - y_offset - i*padding for i in range(max_rows_per_page + 1)]

    for rows in grouper(data, max_rows_per_page):
        rows = tuple(filter(bool, rows))
        c.grid(x_list,y_list[:len(rows) + 1])
        for y, row in zip(y_list[:-1], rows):
            for x, cell in zip(x_list, row):
                c.drawString(x + 2, y - padding + 3, str(cell))
        c.showPage()
    c.drawImage('Resultado_' +name, 80, h - 50, width = 490, height = 700)
    c.save()


def documento(results, largo, image_file):
    data = [("Arandano", "Color","Calibre")]
    lista_color = []
    lista_calibre = []
    v = 0
    r_a = 0
    r = 0
    p = 0
    o_e = 0
    calibre_1 = 0
    calibre_2 = 0
    calibre_3 = 0
    calibre_4 = 0
    calibre_5 = 0
    calibre_6 = 0
    calibre_7 = 0
    for i in range(largo):
        color_class = results[i]["color_class"]
        size_x = results[i]["size_x"]
        size_y = results[i]["size_y"]
        if (color_class == 0):
            color = "Verde"
            v = v + 1
        if (color_class == 1):
            color = "Rojo punta Amarillo"
            r_a = r_a + 1
        if (color_class == 2):
            color = "Rojo"
            r = r + 1
        if (color_class == 3):
            color = "Purpura"
            p = p + 1
        if (color_class == 4):
            color = "Optimo Exportacion"
            o_e = o_e + 1
        size = ((size_x + size_y)/2)
        size = round((size * 10),2)
        if (size < 12):
            calibre_1 = calibre_1 + 1
        if (size < 14 and size >= 12):
            calibre_2 = calibre_2 + 1
        if (size < 16 and size >=14):
            calibre_3 = calibre_3 + 1
        if (size < 18 and size >=16):
            calibre_4 = calibre_4 + 1
        if (size < 20 and size >=18):
            calibre_5 = calibre_5 + 1
        if (size < 22 and size >=20):
            calibre_6 = calibre_6 + 1
        if (size >= 22):
            calibre_7 = calibre_7 + 1
        data.append((f"Arandano {i+1}", color, f" {size} mm"))
    lista_color.append(round(((v * 100)/largo),2))
    lista_color.append(round(((r_a * 100)/largo),2))
    lista_color.append(round(((r * 100) / largo),2))
    lista_color.append(round(((p * 100) / largo),2))
    lista_color.append(round(((o_e * 100) / largo),2))
    lista_calibre.append(round(((calibre_1 * 100) / largo),2))
    lista_calibre.append(round(((calibre_2 * 100) / largo),2))
    lista_calibre.append(round(((calibre_3 * 100) / largo),2))
    lista_calibre.append(round(((calibre_4 * 100) / largo),2))
    lista_calibre.append(round(((calibre_5 * 100) / largo),2))
    lista_calibre.append(round(((calibre_6 * 100) / largo),2))
    lista_calibre.append(round(((calibre_7 * 100) / largo),2))
    export_to_pdf(data,image_file,lista_color,lista_calibre)


largo = len(frame_cereza2)
#print (largo)
documento(frame_cereza2, largo, image_file)

