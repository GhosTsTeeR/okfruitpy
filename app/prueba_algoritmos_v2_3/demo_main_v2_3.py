from prueba_algoritmos_v2_3.declarar_librerias import *
from prueba_algoritmos_v2_3.declarar_variables import *
from prueba_algoritmos_v2_3.declarar_rutas import *

def inicializar_arandanos(ruta_img, nombre_archivo, numero_aleatorio, fecha_actual, username):
    ruta_guardar= 'guardar_analisis/'
    img = cv2.imread(ruta_img+nombre_archivo)
    tam_y, tam_x, _ = img.shape

    #se hacen copias de las imagenes para los reportes
    original = img.copy()
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
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

    _, contours, hierachy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    tamanio_w = 0
    tamanio_h = 0
    cantidad = 0
    #Verificación para eliminar ruido de la imagen (Contornos pequeños)
    for index, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        tamanio_w = tamanio_w + w
        tamanio_h = tamanio_h + h
        cantidad = cantidad + 1

    tamanio_w = tamanio_w / cantidad
    tamanio_h = tamanio_h / cantidad
    cantidad = 0
    area_prom = 0

    #Verificación para eliminar ruido de la imagen (Contornos pequeños)
    for index, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        if (w > tamanio_w and h > tamanio_h):
            cantidad = cantidad + 1
            area = cv2.contourArea(cnt)
            area_prom = area_prom + area

    area_prom = area_prom / cantidad
    cantidad = 0
    prom_w = 0
    prom_h = 0

    # print (tamanio_w,tamanio_h)
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


    #Verificación de los contornos para encontrar los frutos
    contador_fruto = 1
    numero_fruto = 1
    for index, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(output,(x,y),(x+w,y+h),(255,255,0),2)
        #Se define el radio de las circunferencias a analizar 
        area = cv2.contourArea(cnt)

        if ((w*h > area_prom/2) and (w*h > 12000) and (tam_y > 3000 or tam_x > 3000)):
            bandera = True
        elif ((w*h > area_prom/2) and (w*h > 1000) and (tam_y < 3000 or tam_x < 3000)):
            bandera = True 
        elif ((w*h > area_prom/2) and (w*h > 1000) and (tam_y < 1000 or tam_x < 1000)):
            bandera = True
        elif ((w*h > area_prom*0.95) or (w*h > 1000) or (tam_y < 1000 or tam_x < 1000)): ## permite procesar arandanos pequeños.
            bandera = True
        else:
            bandera = False

        if bandera:
            # if ((w*h)  > area_prom/2):
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
                    w_, h_, radius, angulo = calibre_2(roi, roi_mask, numero_fruto, name_file, ruta_guardar)       
                    roi_list.append(roi)
                    color_class = getClass(roi,roi_mask)
                    porcentaje_a_rojizo, porcentaje_a_claro, porcentaje_a_optimo, porcentaje_negro = segmentamos_color(roi,roi_mask)
                    
                    pixeles_sum = calculate_pixeles_sum(roi, ruta_guardar, numero_fruto, name_file)
                    porcentaje_danio, porcentaje_manipulacion = danio_pudricion_2_2(roi,ruta_guardar,numero_fruto,name_file, pixeles_sum)
                    
                    pos_i, pos_j = ubicacion2(tam_x,tam_y,x,y,w_,h_)
                    numero_fruto = numero_fruto + 1
                    results.append({
                        'x'           : x,
                        'y'           : y,
                        'w'           : w_,
                        'h'           : h_,
                        'size_x'      : (w_ * constante_mm) / dpi_x, 
                        'size_y'      : (h_ * constante_mm) / dpi_y,
                        'size'        : 0,
                        'porcentaje_a_rojizo' : porcentaje_a_rojizo,
                        'porcentaje_a_claro' : porcentaje_a_claro,
                        'porcentaje_a_optimo' : porcentaje_a_optimo,
                        'porcentaje_negro' : porcentaje_negro,
                        'color_class' : color_class,
                        'danio'       : porcentaje_danio,
                        'manipulacion': porcentaje_manipulacion,
                        'radius'      : radius,
                        'ubicacion_i' : pos_i,
                        'ubicacion_j' : pos_j, 
                    })

    # for r in frame_cereza2:
    numero_fruto = 1
    for r in results:
        x = int(r['x'])
        y = int(r['y'])
        w = int(r['w'])
        h = int(r['h'])
    
        cv2.rectangle(original, (x,y), (x+w,y+h), (255,255,0), 2)
        cv2.putText(original, str(numero_fruto), (x,y), font, 1.8, (255,0,0), 3, cv2.LINE_AA) #Aqui puedes cambiar el tamaño de los números de 0.7 a 1.5 recomendado y el grosor de 2 a 3
        numero_fruto = numero_fruto + 1

    img_result = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    cv2.imwrite(ruta_guardar + "img" + numero_aleatorio + version + fecha_actual + ".jpg", img_result)

    # largo = len(frame_cereza2)
    largo = len(results)
    documento(results, largo, image_file, ruta_guardar, name_file, h, numero_aleatorio, fecha_actual, username)
