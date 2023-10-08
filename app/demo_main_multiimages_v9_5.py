from declarar_librerias import *
from declarar_rutas_multiimagenes import *

for image in glob.glob(image_file):
    #Variables globales
    font = cv2.FONT_HERSHEY_SIMPLEX
    numero_fruto = 1
    referencia_imagen = 0
    cantidad_frutos_max = 100

    #Variables globales para evitar el error de segmentación
    tamanio_w = 0
    tamanio_h = 0
    cantidad = 0
    area_prom = 0
    prom_w = 0
    prom_h = 0
    cantidad_ima = 0

    ### 
    roi_list  = []
    hist_list = []
    results = []

    valor_muestra = 1
    muestra = "Muestra_"

    kernel = np.ones((5,5), np.uint8)

    # defecto: 300, 300
    # android (silvia): 320, 320
    # iphone 11: 324, 324

    ## buenos resultados con 200
    ## buenos resultados con 240

    ## imagenes en terreno (05-12-2021) dpi = 290 y cant. iteraciones = 1
    pixel_inch = 210 # 300 # 290 # 260 # 240 # 220 # 200

    ## con cantidad_iteraciones = 1 segmenta justo el fruto.
    cantidad_iteraciones = 1 # 4 # 3 # 2 # 1
    constante_mm = 26.4583333337192 # 25.4
    dpi_x, dpi_y = pixel_inch, pixel_inch # 300, 300 # 425, 425

    version = "_v9_5_"
    pixel_negro = 0
    pixel_uno = 1

    img = None

    marco = 27 # 31

    Categories = []
    Categories.append("Fruta sana")
    Categories.append("Fruta dañada")

    name_file = image.split("/")[-1]
    
    ##
    t_0 = time.process_time_ns()
    
    ##
    img = cv2.imread(image)
    print(name_file)

    cv2.imwrite(ruta_guardar+"img_original" + version + name_file, img)

    #se hacen copias de las imagenes para los reportes
    original = img.copy()
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    ## EVALUAR OPTIMIZAR RESULTADOS MEJORANDO ESTE FILTRO.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    ret2, th = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    mask1 = cv2.bitwise_and(img, img, mask = th)
    white = np.zeros_like(img)
    white = cv2.bitwise_not(white)
    mask2 = cv2.bitwise_and(white, white, mask = cv2.bitwise_not(th))
    img = mask2 + mask1

    ## Canal HSV
    img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

    # #pasar por filtro para eliminar pedicelo
    mask_verde = cv2.inRange(img_hsv,(35,20,20),(65,255,255)) ## Excelentes resultados.
    # cv2.imwrite(ruta_guardar+"mask_verde" + version + name_file, mask_verde)

    mask_verde_2 = cv2.inRange(img_hsv,(95,56,36),(98,81,40)) ## Excelentes resultados.
    # cv2.imwrite(ruta_guardar+"mask_verde_2" + version + name_file, mask_verde_2)

    mask_verde_3 = cv2.inRange(img_hsv,(25,52,72),(102,255,255))
    # cv2.imwrite(ruta_guardar+"mask_verde_3_"+str(numero_fruto) + version + name_file, mask_verde_3)

    mask_verde_4 = cv2.inRange(img_hsv,(21,149,24),(102,255,255))
    # cv2.imwrite(ruta_guardar+"mask_verde_4_"+str(numero_fruto) + version + name_file, mask_verde_4)

    mask_verde_5 = cv2.inRange(img_hsv,(4,130,13),(102,255,255))
    # cv2.imwrite(ruta_guardar+"mask_verde_5_"+str(numero_fruto) + version + name_file, mask_verde_5)

    # ## Eliminamos pedicelo deshidratado (color cafe verdoso).
    mask_deshidratado = cv2.inRange(img_hsv,(30,25,25),(95,255,255)) ## Buenos resultados.
    # cv2.imwrite(ruta_guardar+"mask_deshidratado" + version + name_file, mask_deshidratado)

    mask_deshidratado_2 = cv2.inRange(img_hsv,(109,57,17),(160,92,33)) ## Buenos resultados.
    # # cv2.imwrite(ruta_guardar+"mask_deshidratado_2" + version + name_file, mask_deshidratado_2)

    mask_deshidratado_3 = cv2.inRange(img_hsv,(100,50,10),(160,95,35)) ## Buenos resultados.
    # # cv2.imwrite(ruta_guardar+"mask_deshidratado_3" + version + name_file, mask_deshidratado_3)

    mask_deshidratado_4 = cv2.inRange(img_hsv,(110,58,17),(160,120,33)) ## Buenos resultados.
    # # cv2.imwrite(ruta_guardar+"mask_deshidratado_4" + version + name_file, mask_deshidratado_4)

    mask_bandeja = cv2.inRange(img_hsv,(103,49,50),(107,67,76)) ## Buenos resultados.
    # # cv2.imwrite(ruta_guardar+"mask_bandeja" + version + name_file, mask_bandeja)

    ##
    mask = mask_verde | mask_verde_2 | mask_verde_3 | mask_verde_4 | mask_verde_5 | mask_deshidratado
    # mask = mask_verde | mask_verde_2 | mask_verde_3 | mask_verde_4 | mask_verde_5 | mask_deshidratado | mask_deshidratado_2 | mask_deshidratado_3 | mask_deshidratado_4 | mask_bandeja

    ######################################################################
    ## kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 5)
    # cv2.imwrite(ruta_guardar+"mask" + version + name_file, mask)

    ##
    target = cv2.inpaint(img, mask, 2, cv2.INPAINT_NS)
    # cv2.imwrite(ruta_guardar+"target" + version + name_file, target)

    # Mascara para ignorar el fondo blanco
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    mask = mask // 255

    #eliminar el fondo blanco
    b,g,r = cv2.split(target)
    b = b * mask
    g = g * mask 
    r = r * mask
    relevant = cv2.merge((b,g,r)).astype(np.uint8)
    # cv2.imwrite(ruta_guardar+"relevant" + version + name_file, relevant)


    ## Removemos el brillo
    # clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))

    # # convert to gray
    # gray = cv2.cvtColor(relevant, cv2.COLOR_BGR2GRAY)
    # grayimg = gray

    # GLARE_MIN = np.array([0, 0, 50],np.uint8)
    # GLARE_MAX = np.array([0, 0, 225],np.uint8)

    # hsv_img = cv2.cvtColor(relevant,cv2.COLOR_BGR2HSV)

    # #HSV
    # frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)

    # #INPAINT
    # mask1 = cv2.threshold(grayimg , 220, 255, cv2.THRESH_BINARY)[1]
    # result1 = cv2.inpaint(relevant, mask1, 0.1, cv2.INPAINT_TELEA) 

    # #CLAHE
    # claheCorrecttedFrame = clahefilter.apply(grayimg)

    # #COLOR 
    # lab = cv2.cvtColor(relevant, cv2.COLOR_BGR2LAB)
    # lab_planes = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    # lab_planes[0] = clahe.apply(lab_planes[0])
    # lab = cv2.merge(lab_planes)
    # clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # #INPAINT + HSV
    # result = cv2.inpaint(relevant, frame_threshed, 0.1, cv2.INPAINT_TELEA) 

    # #INPAINT + CLAHE
    # grayimg1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    # mask2 = cv2.threshold(grayimg1 , 220, 255, cv2.THRESH_BINARY)[1]
    # result2 = cv2.inpaint(relevant, mask2, 0.1, cv2.INPAINT_TELEA) 

    # #HSV+ INPAINT + CLAHE
    # lab1 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    # lab_planes1 = cv2.split(lab1)
    # clahe1 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    # lab_planes1[0] = clahe1.apply(lab_planes1[0])
    # lab1 = cv2.merge(lab_planes1)
    # relevant_sin_brillo = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)
    # cv2.imwrite(ruta_guardar+"relevant_sin_brillo" + version + name_file, relevant_sin_brillo)

    # ## 
    # # threshold grayscale image to extract glare
    # relevant_gray = cv2.cvtColor(relevant, cv2.COLOR_BGR2GRAY)
    # relevant_mask = cv2.threshold(relevant_gray, 220, 255, cv2.THRESH_BINARY)[1]

    # # use mask with input to do inpainting
    # relevant_sin_brillo = cv2.inpaint(relevant, relevant_mask, 21, cv2.INPAINT_TELEA) 
    # cv2.imwrite(ruta_guardar+"relevant_sin_brillo" + version + name_file, relevant_sin_brillo)


    #separar los canales
    b,g,r = cv2.split(relevant)
    mix = 0.9*r+0.1*g
    mix = mix.astype(np.uint8)

    #contorno
    ret, thresh = cv2.threshold(mix,1,255,cv2.THRESH_BINARY)
    ## kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.erode(thresh, kernel, iterations = cantidad_iteraciones) 
    # cv2.imwrite(ruta_guardar+"thresh" + version + name_file, thresh)

    ## Detecta contornos, sobre target (imagen sin fondo, sin pedicelos, solo fruto y fondo blanco)
    _, contours, hierachy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
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

    if(cantidad != 0):
        prom_w = (prom_w / cantidad)
        prom_h = (prom_h / cantidad)

    #Verificación de los contornos para encontrar los frutos
    contador_fruto = 0
    contador = 0
    for index, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(output,(x,y),(x+w,y+h),(255,255,0),2)

        if (w > tamanio_w and h > tamanio_h and (w*h) > (tamanio_h*tamanio_w) and (w>112 != h>112)):
            area = cv2.contourArea(cnt)
            if (((w*h)  > area_prom/2) and (w*h < 99900)):
                if (w > h):
                    diferencia = w - h
                    valor = h
                if (h >= w):
                    diferencia = h - w
                    valor = w
                if (diferencia < valor):
                    # if (w < prom_w*1.5 and h < prom_h*1.5):
                    if (w < prom_w*1.5 and h < prom_h*1.5 and y-marco > 0 and y+h+marco > 0 and x-marco > 0 and x+w+marco > 0):
                        print("Fruto: ", numero_fruto)
                        roi = original[y:y+h, x:x+w]
                        # cv2.imwrite(ruta_guardar+"roi_"+str(numero_fruto) + version + name_file, roi)

                        roi_mask = thresh[y:y+h, x:x+w] # // 255
                        # cv2.imwrite(ruta_guardar+"roi_mask_"+str(numero_fruto) + version + name_file, roi_mask)
                        
                        ##
                        w_, h_, radius, angulo = calibre_2(roi, roi_mask, numero_fruto, name_file, ruta_guardar)
                        roi_mask = roi_mask // 255
                        color_class = getClass(roi,roi_mask)
                        porcentaje_rojo_claro, porcentaje_rojo, porcentaje_rojo_caoba, porcentaje_caoba_oscuro, porcentaje_negro = segmentamos_color(roi,roi_mask)

                        roi_marco = original[y-marco:y+h+marco, x-marco:x+w+marco]
                        # cv2.imwrite(ruta_guardar+"roi_marco_"+str(numero_fruto) + version + name_file, roi_marco)

                        pedicelo = deteccion_pedicelo(roi_marco, ruta_guardar, numero_fruto, version, name_file)

                        roi_sin_pedicelo = relevant[y:y+h,x:x+w]
                        ## roi_sin_pedicelo = relevant_sin_brillo[y:y+h,x:x+w]
                        # cv2.imwrite(ruta_guardar+"roi_sin_pedicelo_"+str(numero_fruto) + version + name_file, roi_sin_pedicelo) #cv2.cvtColor(roi_sin_pedicelo, cv2.COLOR_RGB2BGR))

                        # ## funcion danio_pudricion filtra por rangos de color danio, pudricion o deshidratado, validar con imagenes con este tipo de rangos.
                        # danio_pudri_1 = danio_pudricion_1(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file)
                        
                        pixeles_sum = calculate_pixeles_sum(roi_sin_pedicelo, ruta_guardar, numero_fruto, name_file)

                        ## funcion danio_pudri_2 filtra por rangos de colores del fruto y resta al total de la imagen, la diferencia es danio, pudricion y deshidratacion, pero tambien puede ser pedicelo.
                        danio_pudri_2  = danio_pudricion_2(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file, pixeles_sum)
                        
                        # danio_pudri_2_1 = danio_pudricion_2_1(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file, pixeles_sum)
                        # danio_pudri_2_2 = danio_pudricion_2_2(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file, pixeles_sum)
                        danio_pudri_2_3 = danio_pudricion_2_3(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file, pixeles_sum)
                        # danio_pudri_2_4 = danio_pudricion_2_4(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file, pixeles_sum)
                        
                        ##
                        # danio_pudricion_3(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file)
                        # danio_pudricion_3_1(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file)

                        ## Usar estos tres metodos para determinar el danio, pudricion y deshidratacion
                        ## funcion danio_pudricion_4 el "blanco" es danio, pudricion y deshidratacion, pero tambine puede ser sombra o pedicelo.
                        # danio_pudricion_4(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file)
                        
                        ## funcion danio_pudricion_5 el "negro" es danio, pudricion y deshidratacion, pero tambien puede ser sombra o pedicelo.
                        # danio_pudricion_5(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file)
                        
                        ## funcion danio_pudricion_6 las "partes sombreadas" son danio, pudricion y deshidratacion, pero tambien puede ser sombra o pedicelo.
                        # danio_pudricion_6(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file)

                        ## Usamos funciones SLIC, SEEDS y LSC
                        # danio_pudricion_7(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file)

                        # print("Fruto: ", numero_fruto, "Danio 1: ", danio_pudri, "Danio 2: ", danio_pudri_2)
                        # print("Fruto: ", numero_fruto, "Danio 1: ", danio_pudri, "Danio 2: ", danio_pudri_2, "Danio 3: ", contrast, dissimilarity, homogeneity, energy, correlation, ASM)

                        # fruta_sana, fruta_danada = danio_pudricion_8(roi_sin_pedicelo,ruta_guardar,numero_fruto,name_file)

                        # if(danio_pudri<danio_pudri_2):
                        #   danio = danio_pudri
                        # else:
                        #   danio = danio_pudri_2

                        # ## pos_i,pos_j = ubicacion2(tam_x,tam_y,x,y,w,h)

                        ## calculamos danoi restando los danios 2 y 2_3
                        danio = round(danio_pudri_2 - danio_pudri_2_3,1)
                        
                        numero_fruto = numero_fruto + 1 
                        
                        results.append({
                            'x'           : x,
                            'y'           : y,
                            'w'           : w_,
                            'h'           : h_,
                            'size_x'      : (w_ * constante_mm) / dpi_x, 
                            'size_y'      : (h_ * constante_mm) / dpi_y,
                            'posicion'    : contador_fruto, 
                            'color_class' : color_class,
                            'porcentaje_rojo_claro' : porcentaje_rojo_claro,
                            'porcentaje_rojo' : porcentaje_rojo,
                            'porcentaje_rojo_caoba' : porcentaje_rojo_caoba, 
                            'porcentaje_caoba_oscuro' : porcentaje_caoba_oscuro,
                            'porcentaje_negro' : porcentaje_negro,
                            'danio'   : danio,
                            # 'danio_pudri_1' : danio_pudri_1,
                            # 'danio_pudri_2' : danio_pudri_2,
                            # 'danio_pudri_2_1' : danio_pudri_2_1,
                            # 'danio_pudri_2_2' : danio_pudri_2_2,
                            # 'danio_pudri_2_3' : danio_pudri_2_3,
                            # 'danio_pudri_2_4' : danio_pudri_2_4,
                            # 'fruta_sana' : fruta_sana,
                            # 'fruta_danada' : fruta_danada,
                            'pedicelo'    : pedicelo,
                            # 'ubicacion_i' : pos_i,
                            # 'ubicacion_j' : pos_j,
                            'angulo'      : angulo 
                        })

    if results:
        numero_fruto = 1
        for r in results:
            x = r['x']
            y = r['y']
            w = r['w']
            h = r['h']

            cv2.rectangle(original, (x,y), (x+w,y+h), (255,255,0), 2)
            cv2.putText(original, str(numero_fruto), (x,y), font, 1.8, (255,0,0), 3, cv2.LINE_AA) #Aqui puedes cambiar el tamaño de los números de 0.7 a 1.5 recomendado y el grosor de 2 a 3

            numero_fruto = numero_fruto + 1

        img_result = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        cv2.imwrite(ruta_guardar+"Resultado" + version + name_file, img_result)

        largo = len(results)
        documento(results, largo, h, ruta_guardar, name_file, image_file, tam_y, tam_x)

    else:
        json_error(ruta_guardar,name_file)
        print("Imagen no se proceso, tome una nueva imagen")

    print("Tiempo: " + str((time.process_time_ns() - t_0)/1000000000) + " seg.")

