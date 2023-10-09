from prueba_algoritmos_v9_5.declarar_librerias import * 

## Colores escala LAB
colors = []
# colors.append((32 , 41 ,-70))  #rojo claro
colors.append((6.91, 11.35, 4.15)) ## Excelentes resultados.

# colors.append((32, 30, -30)) #rojo ## POR MEJORAR
# colors.append((4.75, 15.28, 3.03)) ## buenos resultados.
colors.append((5, 15.28, 3.03)) ## Muy buenos resultados.

# colors.append((32, 17, -15))  #rojo caoba
# colors.append((3,4,0)) ## Buenos resultados
# colors.append((3.89,4.58,0.93)) ## Buenos resultados
# colors.append((3.89,5.58,0.93)) ## Muy Buenos Resultados.
# colors.append((3.89,6.58,0.93)) ## Muy Buenos Resultados.
# colors.append((3.89,7.58,0.93)) ## Muy Buenos Resultados. 
# colors.append((4.89,7.58,0.93)) ## Muy Buenos Resultados.
colors.append((5.89,7.96,-2.86)) ## Muy Buenos Resultados.

# # colors.append((32, 8, -8))   #caoba
# colors.append((4,1,-4))

# colors.append((32, 4, -5))    #caoba oscuro
colors.append((3, 1, -3)) ## Excelente resultados
# colors.append((2.66, 0.82, -1.9)) ## Excelente resultados
# colors.append((2.49, 0.87, -1.73))

# colors.append((32, 2, 0))    #negro ## PENDIENTE
# colors.append((3,0,-2))
# colors.append((1.56, 0.57, -1.23))
# colors.append((0.56, -1.57, -2.23))
colors.append((0.56, 0, -5))

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
    for x in range(0,6):
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
    clase = establecer_color(distancias)
    return clase


#image = cv2.imread('5_1_1.png')
#mask = getMask(image)
#distancias = similitud_color(image,mask)
#print (distancias)
#clase = establecer_color(distancias)
#print ("La imagen correponde a la clase", clase+1)
