from skimage import color
from skimage import io
from skimage.util.dtype import dtype_range
from skimage import exposure
from skimage.color import rgb2lab, lab2lch
import cv2
import numpy as np
import matplotlib.pyplot as plt

colors = []
# colors.append((33,-24,0))  #verde
# colors.append((40,-14,-16)) #rojo_amarillo_parte_roja
# colors.append((15,15,-15))  #rojo
# colors.append((27,7,6))    #morado o purpura

# colors.append((5.32595, 2.52, -3.12)) # A. R. o A Rojizo
colors.append((27.71, 7.69, -8.85)) ## Leves resultados.

# colors.append((6.27, 0.20, -3.51)) # A. CL o Claro
colors.append((28.18, 0.31, -10.21)) ## Buenos resultados
# colors.append((31.02, -0.93, -8.44))

# colors.append((31,6,-2))   #purpura o optimo exportacion
# colors.append((3.61, 0.29, -3.40)) # Resultados leves
colors.append((9.4, 1.9, -8.82)) ## Buenos resultados
# colors.append((22, 26, 38))
# colors.append((26, 31.9, 47.9))

#colors.append((6,-2,-3))    #morado

# colors.append((1.49144, 0.06, -1.13)) # negro
colors.append((2.484, 0.177, -0.803)) ## Buenos resultados

def distancia_color(imagen_lab, color_id):
    return color.deltaE_ciede2000(imagen_lab,colors[color_id])

def analisis_distancia(matriz, dimensiones, largo, mask):
    elementos = 0
    cantidad = 0
    for j in range(0,largo):
        for x in range(0,dimensiones):
            if (mask[j][x] != 0):
               elementos = elementos + matriz[j][x]*mask[j][x]
               cantidad = cantidad + 1
    elementos = elementos / cantidad
    return elementos

def similitud_color(imagen, mask):
    imagen_lab = rgb2lab(imagen)
    #print(imagen_lab)
    dimensiones = len(imagen[0])
    distancia = []
    for i in range(len(colors)):
        matriz = distancia_color(imagen_lab, i)
        largo = len(matriz)
        distancia.append(analisis_distancia(matriz,dimensiones,largo,mask))
    return distancia

def establecer_color(distancia):
    deltaE = 100
    resultado = -1
    for i in range(len(distancia)):
        if distancia[i] < deltaE:
            deltaE = distancia[i]
    for i in range(len(distancia)):
        if distancia[i] == deltaE:
            resultado = i
    return resultado

def getMask(image):
    image = cv2.medianBlur(image,7)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    mask = mask // 255
    return mask

def getClassFromImage(image, mask, colors_template=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_colors = []
    for x in range(0,5):
        col = [[[x[0], x[1], x[2]]]]
        npcol = np.array(col, dtype=np.uint8)
        converted = cv2.cvtColor(npcol, cv2.COLOR_BGR2LAB)
        converted = [converted[0][0][0]*(100/255), converted[0][0][1]-128, converted[0][0][2]-128]
        new_colors.append(converted)
    colors = new_colors
    distancias = similitud_color(image, mask)
    clase = establecer_color(distancias)
    return clase
    
def getClass(image,mask):
    #mask = getMask(image)
    distancias = similitud_color(image,mask)
    #print (distancias)
    clase = establecer_color(distancias)
    return clase


#image = cv2.imread('negro.png')
#mask = getMask(image)
#distancias = similitud_color(image,mask)
#print (distancias)
#clase = establecer_color(distancias)
#print ("La imagen correponde a la clase", clase+1)


