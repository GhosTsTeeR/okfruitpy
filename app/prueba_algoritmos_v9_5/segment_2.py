import numpy as np
import cv2
import argparse
from reportlab.pdfgen import canvas
from prueba_algoritmos_v9_5.extraer_pixel import *
from prueba_algoritmos_v9_5.color_ import *
from prueba_algoritmos_v9_5.square_3 import square
from prueba_algoritmos_v9_5.palito import *
from prueba_algoritmos_v9_5.pudricion import *
import pandas as pd
import random
import itertools
from reportlab.lib.pagesizes import A4
import matplotlib.pyplot as plt
from prueba_algoritmos_v9_5.posicion import * 


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

#leer la imagen
parser = argparse.ArgumentParser(description='Image read.')
parser.add_argument('--image', '-i',help='Image to open.')
args = parser.parse_args()
name = args.image
image_file = args.image
img = cv2.imread(image_file)

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

#Prueba para eliminar el ruido
#img = cv2.fastNlMeansDenoisingColored(img,None,20,10,7,21)

#pasar por filtro para eliminar palitos
mask_verde = cv2.inRange(original_hsv,(32,0,0),(65,255,255)) #este filtro se debe calibrar con la caja Verde
mask_amarilla = cv2.inRange(original_hsv,(15,0,0),(36,255,255)) #este filtro se debe calibrar con la caja Amarillo
mask_not = cv2.add(mask_amarilla,mask_verde)
target = cv2.bitwise_not(img,img,mask=mask_verde)
#plt.imshow(target)
#plt.show()

# Mascara para ignorar el fondo blanco
gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
mask = mask // 255

#eliminar el fondo blanco
b,g,r = cv2.split(target)
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

_, contours, hierachy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()


roi_list  = []
hist_list = []

results = []
tam_y,tam_x,_ = img.shape

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
img_square, cX, cY = square(img)
print (cX,cY)
#el cuadrado mide 5cm se escala para saber el tamaño de los pixeles
cX =  (float(5/cX))
cY =  (float(5/cY))
#Verificación de los contornos para encontrar los frutos
contador_fruto = 0
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
           if (w < prom_w*1.5 and h < prom_h*1.5):
                contador_fruto = contador_fruto + 1
                roi = img[y:y+h, x:x+w]
                roi_mask = thresh[y:y+h, x:x+w] // 255
                roi_list.append(roi)
                color_class = getClass(roi,roi_mask)
                danio,imagen_pixel = danio_palito(roi)
                danio_pudri = danio_pudricion(roi)
                pos_i,pos_j = ubicacion(tam_x,tam_y,x,y,w,h)
                results.append({
                   'x'           : x,
                   'y'           : y,
                   'w'           : w,
                   'h'           : h,
                   'size_x'      : w*cX,
                   'size_y'      : h*cY,
                   'posicion'    : contador_fruto, 
                   'color_class' : color_class,
                   'diametro'      : radius*2,
                   'pudricion'   : danio_pudri,
                   'palito'      : danio,
                   'ubicacion_i' : pos_i })

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

# print (lista_contadores)
if (len(lista_contadores) == 5):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5])

if (len(lista_contadores) == 6):
	d1 = data_frame[0:lista_contadores[0]].sort_values('x')
	d2 = data_frame[lista_contadores[0]:lista_contadores[1]].sort_values('x')
	d3 = data_frame[lista_contadores[1]:lista_contadores[2]].sort_values('x')
	d4 = data_frame[lista_contadores[2]:lista_contadores[3]].sort_values('x')
	d5 = data_frame[lista_contadores[3]:lista_contadores[4]].sort_values('x')
	d6 = data_frame[lista_contadores[4]:lista_contadores[5]].sort_values('x')
	frame_cereza = pd.concat([d1,d2,d3,d4,d5,d6])
	
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

if (len(lista_contadores) >= 12):
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
cv2.imwrite("Resultado_2_" + image_file, img_result)


def grouper(iterable,n):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args)

def export_to_pdf(data,image_file,lista_color,lista_calibre,lista_pedicelo):
    c = canvas.Canvas("Reporte_Cerezas_2_"+image_file+".pdf", pagesize=A4)
    c.drawString(400,800, "Reporte General Cerezas")
    c.drawString(120,750, "Resumen de color")
    for i in range(0,4):
        if (lista_color[0] >= 0):
           c.drawString(100,700, "Porcentaje de Cerezas de Color Rojo Claro : "+ str(lista_color[0])+" %")
        if (lista_color[1] >= 0):
           c.drawString(100,680, "Porcentaje de Cerezas de Color Rojo : "+ str(lista_color[1])+" %")
        if (lista_color[2] >= 0):
           c.drawString(100,660, "Porcentaje de Cerezas de Color Rojo Caoba : "+ str(lista_color[2])+" %")
        if (lista_color[3] >= 0):
           c.drawString(100,640, "Porcentaje de Cerezas de Color Caoba : "+ str(lista_color[3])+" %")
        if (lista_color[4] >= 0):
           c.drawString(100,620, "Porcentaje de Cerezas de Color Caoba Oscuro : "+ str(lista_color[4])+" %")
        if (lista_color[5] >= 0):
           c.drawString(100,600, "Porcentaje de Cerezas de Color Negro : "+ str(lista_color[5])+" %")
    c.drawString(120,570, "Resumen de Calibre (mm) ")
    for i in range(0,4):
        if (lista_calibre[0] >= 0):
           c.drawString(100,530, "Porcentaje de Cerezas con tamaño inferior a 22 mm : "+ str(lista_calibre[0])+" %")
        if (lista_calibre[1] >= 0):
           c.drawString(100,510, "Porcentaje de Cerezas con tamaño entre 22 y 24 mm : "+ str(lista_calibre[1])+" %")
        if (lista_calibre[2] >= 0):
           c.drawString(100,490, "Porcentaje de Cerezas con tamaño entre 24 y 26 mm : "+ str(lista_calibre[2])+" %")
        if (lista_calibre[3] >= 0):
           c.drawString(100,470, "Porcentaje de Cerezas con tamaño entre 26 y 28 mm : "+ str(lista_calibre[3])+" %")
        if (lista_calibre[4] >= 0):
           c.drawString(100,450, "Porcentaje de Cerezas con tamaño entre 28 y 30 mm : "+ str(lista_calibre[4])+" %")
        if (lista_calibre[5] >= 0):
           c.drawString(100,430, "Porcentaje de Cerezas con tamaño superior a 30 mm : "+ str(lista_calibre[5])+" %")
    c.drawString(120,400, "Resumen de Cerezas con o sin Pedicelo")
    for i in range(0,4):
        if (lista_pedicelo[0] >= 0):
            c.drawString(100,370, "Porcentaje de Cerezas con Pedicelo: "+ str(lista_pedicelo[0])+" %")
        if (lista_pedicelo[1] >= 0):
            c.drawString(100,350, "Porcentaje de Cerezas sin Pedicelo: "+ str(lista_pedicelo[1])+" %")
    c.showPage()
    a, b = A4
    max_rows_per_page = 45
    x_offset = 50
    y_offset = 80
    padding = 15
    x_list = [ x + x_offset for x in [0,100,200,300,400,500]]
    y_list = [ b - y_offset - i*padding for i in range(max_rows_per_page + 1)]
    for rows in grouper(data, max_rows_per_page):
        rows = tuple(filter(bool, rows))
        c.grid(x_list,y_list[:len(rows) + 1])
        for y, row in zip(y_list[:-1], rows):
            for x, cell in zip(x_list, row):
                c.drawString(x + 2, y - padding + 3, str(cell))
        c.showPage()
    #c.drawImage('Resultado_cereza_4.jpeg', 80, h - 50, width = 490, height = 700)
    c.drawImage('Resultado_2_' +name, 80, h -50, width = 490, height = 700)
    
    c.save()


def documento(results, largo, image_file):
    data = [("Cerezas", "Color","Calibre","Pedicelo","Danio")]
    lista_color = []
    lista_calibre = []
    lista_pedicelo = []
    v = 0
    r_a = 0
    r = 0
    p = 0
    o_e = 0
    N = 0
    calibre_1 = 0
    calibre_2 = 0
    calibre_3 = 0
    calibre_4 = 0
    calibre_5 = 0
    calibre_6 = 0
    positivo = 0
    negativo = 0
    for i in range(largo):
        color_class = results[i]["color_class"]
        size_x = results[i]["size_x"]
        size_y = results[i]["size_y"]
        diametro = results[i]["diametro"]
        danio = results[i]["pudricion"]
        palito = results[i]["palito"]
        ubicacion_i = results[i]["ubicacion_i"]
        danio = round(danio,2)
        if (color_class == 0):
            color = "Rojo Claro"
            v = v + 1
        if (color_class == 1):
            color = "Rojo "
            r_a = r_a + 1
        if (color_class == 2):
            color = "Rojo Caoba"
            r = r + 1
        if (color_class == 3):
            color = "Caoba"
            p = p + 1
        if (color_class == 4):
            color = "Caoba Oscuro"
            o_e = o_e + 1
        if (color_class == 5):
            color = "Negro"
            N = N + 1
        if (size_x > size_y):
            size = size_x
        if (size_y >= size_x):
            size = size_y
        
        if (ubicacion_i == 0): size = size*0.95
        if (ubicacion_i == 1): size = size*0.965
        if (ubicacion_i == 5): size = size*0.965
        if (ubicacion_i == 6): size = size*0.95
        #size = ((size_x + size_y)/2)
        size = round((size * 10),2)
        #size = round((diametro*c*10),2)
        if (size < 22):
            calibre_1 = calibre_1 + 1
        if (size < 24 and size >= 22):
            calibre_2 = calibre_2 + 1
        if (size < 26 and size >=24):
            calibre_3 = calibre_3 + 1
        if (size < 28 and size >=26):
            calibre_4 = calibre_4 + 1
        if (size < 30 and size >=28):
            calibre_5 = calibre_5 + 1
        if (size >= 30):
            calibre_6 = calibre_6 + 1
        if (palito == "Si"):
            positivo = positivo + 1
        if (palito == "No"):
            negativo = negativo + 1
        #data.append((f"Cereza {i+1}", color, f" {size} mm"))
        data.append(("Cereza " + str(i + 1), color, str(size) + " mm", palito, str(danio) + " %"))
    
    lista_color.append(round(((v * 100)/largo),2))
    lista_color.append(round(((r_a * 100)/largo),2))
    lista_color.append(round(((r * 100) / largo),2))
    lista_color.append(round(((p * 100) / largo),2))
    lista_color.append(round(((o_e * 100) / largo),2))
    lista_color.append(round(((N * 100) / largo),2))
    lista_calibre.append(round(((calibre_1 * 100) / largo),2))
    lista_calibre.append(round(((calibre_2 * 100) / largo),2))
    lista_calibre.append(round(((calibre_3 * 100) / largo),2))
    lista_calibre.append(round(((calibre_4 * 100) / largo),2))
    lista_calibre.append(round(((calibre_5 * 100) / largo),2))
    lista_calibre.append(round(((calibre_6 * 100) / largo),2))
    lista_pedicelo.append(round(((positivo * 100) / largo),2))
    lista_pedicelo.append(round(((negativo * 100) / largo),2))
    export_to_pdf(data,image_file,lista_color,lista_calibre,lista_pedicelo)
    #print (lista_color)

largo = len(frame_cereza2)
#print (largo)
documento(frame_cereza2, largo, image_file)

