from prueba_algoritmos_v9_5.declarar_librerias import *
from prueba_algoritmos_v9_5.declarar_variables import *
from __init__ import add_datos_cerezas
def grouper(iterable,n):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args)

def export_to_pdf(data,image_file,lista_color,lista_calibre,lista_pedicelo, name_file, h, ruta_guardar, tam_y, tam_x, numero_aleatorio, fecha_actual):
    c = canvas.Canvas(ruta_guardar + "pdf" + numero_aleatorio + version + fecha_actual+".pdf", pagesize=A4)
    c.drawString(400,800, "Reporte General Cerezas")
    c.drawString(120,750, "Resumen de color")
    for i in range(0,4):
        if (lista_color[0] >= 0):
           c.drawString(100,700, "Porcentaje de Cerezas de Color Rojo Claro : "+ f"{str(lista_color[0]).replace('.',',')}" + "%")
        if (lista_color[1] >= 0):
           c.drawString(100,680, "Porcentaje de Cerezas de Color Rojo : "+ f"{str(lista_color[1]).replace('.',',')}" + "%")
        if (lista_color[2] >= 0):
           c.drawString(100,660, "Porcentaje de Cerezas de Color Rojo Caoba : "+ f"{str(lista_color[2]).replace('.',',')}" + "%")
        if (lista_color[3] >= 0):
           c.drawString(100,640, "Porcentaje de Cerezas de Color Caoba Oscuro : "+ f"{str(lista_color[3]).replace('.',',')}" + "%")
        if (lista_color[4] >= 0):
           c.drawString(100,620, "Porcentaje de Cerezas de Color Negro : "+ f"{str(lista_color[4]).replace('.',',')}" + "%")
    c.drawString(120,570, "Resumen de Calibre (mm) ")

    for i in range(0,4):
        if (lista_calibre[0] >= 0):
           c.drawString(100,530, "Porcentaje de Cerezas con tamaño inferior a 22 mm : "+ f"{str(lista_calibre[0]).replace('.',',')}" + "%")
        if (lista_calibre[1] >= 0):
           c.drawString(100,510, "Porcentaje de Cerezas con tamaño entre 22 y 24 mm : "+ f"{str(lista_calibre[1]).replace('.',',')}" + "%")
        if (lista_calibre[2] >= 0):
           c.drawString(100,490, "Porcentaje de Cerezas con tamaño entre 24 y 26 mm : "+ f"{str(lista_calibre[2]).replace('.',',')}" + "%")
        if (lista_calibre[3] >= 0):
           c.drawString(100,470, "Porcentaje de Cerezas con tamaño entre 26 y 28 mm : "+ f"{str(lista_calibre[3]).replace('.',',')}" + "%")
        if (lista_calibre[4] >= 0):
           c.drawString(100,450, "Porcentaje de Cerezas con tamaño entre 28 y 30 mm : "+ f"{str(lista_calibre[4]).replace('.',',')}" + "%")
        if (lista_calibre[5] >= 0):
           c.drawString(100,430, "Porcentaje de Cerezas con tamaño superior a 30 mm : "+ f"{str(lista_calibre[5]).replace('.',',')}" + "%")
    c.drawString(120,400, "Resumen de Cerezas con o sin Pedicelo")
    
    for i in range(0,4):
        if (lista_pedicelo[0] >= 0):
            c.drawString(100,370, "Porcentaje de Cerezas con Pedicelo: " + f"{str(lista_pedicelo[0]).replace('.',',')}" + "%")
        if (lista_pedicelo[1] >= 0):
            c.drawString(100,350, "Porcentaje de Cerezas sin Pedicelo: " + f"{str(lista_pedicelo[1]).replace('.',',')}" + "%")

    c.showPage()
    a, b = A4
    max_rows_per_page = 45
    x_offset = 10
    y_offset = 80
    padding = 15

    ## Determinamos el ancho de cada columna de la tabla.
    # x_list = [ x + x_offset for x in [0,60,100,150,220,270,320,380,450,510,570]] ## [0,90,180,270,360,450,540]] #
    x_list = [ x + x_offset for x in [0,90,180,270,360,450]] #
    y_list = [ b - y_offset - i*padding for i in range(max_rows_per_page + 1)]
    for rows in grouper(data, max_rows_per_page):
        rows = tuple(filter(bool, rows))
        c.grid(x_list,y_list[:len(rows) + 1])
        for y, row in zip(y_list[:-1], rows):
            for x, cell in zip(x_list, row):
                c.drawString(x + 2, y - padding + 3, f"{str(cell).replace('.',',')}")
        c.showPage()
    
    ## Escalamos la imagen a pegar en informe de calidad según dimensiones.
    # scale_percent = 20
    # sp_width = int(tam_x * scale_percent / 100)
    # sp_heigh = int(tam_y * scale_percent / 100)
    sp_width, sp_heigh = 560, 650
    ## ES: ruta + nombre_imagen_procesada
    c.drawImage(ruta_guardar + 'resultado' + numero_aleatorio + version + fecha_actual + ".jpg", 20, 20, width = sp_width, height = sp_heigh)
    c.save()

def json_error(ruta_guardar,name_file):
    # creamos json de resultados.
    data_json = {}
    datos_array = []
    datos = {}
    solicitudImagen_id = {}
    imagenProcesada = {}
    resultadoAnalisis = []
    frutos = {}
    
    data_json['respuesta'] = 'Error'        
    datos['solicitudImagen_id'] = 1
    datos['imagenProcesada'] = name_file
    
    datos_array.append(datos)
    datos_array.append(datos)

    data_json['datos'] = []

    frutos['NumeroFruto'] = ""
    frutos['Color'] = ""
    frutos['Calibre'] = ""
    
    frutos['Pedicelo'] = ""
    frutos['Danio'] = ""
    resultadoAnalisis.append(frutos)
    frutos = {}

    datos['resultadoAnalisis'] =  resultadoAnalisis
    datos_array.append(resultadoAnalisis)
    data_json['datos'].append(datos)

    ## Guardamos json
    ## GAMADIEL, ACA ESTA LO QUE BUSCAS
    archivo_json = os.path.join(ruta_guardar+name_file + version +".json")
    with open(archivo_json, 'w') as  json_file:
        json.dump(data_json, json_file)


def guardar_excel(data,ruta_guardar, name_file, numero_aleatorio, fecha_actual):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(ruta_guardar + "excel" + numero_aleatorio + version + fecha_actual+".xlsx")
    worksheet = workbook.add_worksheet()
    
    # Start from the first cell. Rows and columns are zero indexed.
    row = 0

    # Iterate over the data and write it out row by row.
    # for fruto, tamanio, pedicelo, danio, color, color_class, porcentaje_rojo_claro, porcentaje_rojo, porcentaje_rojo_caoba, porcentaje_caoba_oscuro, porcentaje_negro, danio_pudri_1, danio_pudri_2, danio_pudri_2_1, danio_pudri_2_2, danio_pudri_2_3, danio_pudri_2_4 in (data):
    #     worksheet.write(row, 0, fruto)
    #     worksheet.write(row, 1, tamanio)
    #     worksheet.write(row, 2, pedicelo)
    #     worksheet.write(row, 3, danio)
    #     worksheet.write(row, 4, color)
    #     worksheet.write(row, 5, color_class)
    #     worksheet.write(row, 6, porcentaje_rojo_claro)
    #     worksheet.write(row, 7, porcentaje_rojo)
    #     worksheet.write(row, 8, porcentaje_rojo_caoba)
    #     worksheet.write(row, 9, porcentaje_caoba_oscuro)
    #     worksheet.write(row, 10, porcentaje_negro)
    #     worksheet.write(row, 11, danio_pudri_1)
    #     worksheet.write(row, 12, danio_pudri_2)
    #     worksheet.write(row, 13, danio_pudri_2_1)
    #     worksheet.write(row, 14, danio_pudri_2_2)
    #     worksheet.write(row, 15, danio_pudri_2_3)
    #     worksheet.write(row, 16, danio_pudri_2_4)
    #     row += 1

    # for fruto, tamanio, pedicelo, color, danio_pudri_1, danio_pudri_2, danio_pudri_2_1, danio_pudri_2_2, danio_pudri_2_3, danio_pudri_2_4 in (data):
    #     worksheet.write(row, 0, fruto)
    #     worksheet.write(row, 1, tamanio)
    #     worksheet.write(row, 2, pedicelo)
    #     worksheet.write(row, 3, color)
    #     worksheet.write(row, 4, danio_pudri_1)
    #     worksheet.write(row, 5, danio_pudri_2)
    #     worksheet.write(row, 6, danio_pudri_2_1)
    #     worksheet.write(row, 7, danio_pudri_2_2)
    #     worksheet.write(row, 8, danio_pudri_2_3)
    #     worksheet.write(row, 9, danio_pudri_2_4)
    #     row += 1

    for fruto, tamanio, pedicelo, color, danio in (data):
        worksheet.write(row, 0, fruto)
        worksheet.write(row, 1, f"{str(tamanio).replace('.',',')}")
        worksheet.write(row, 2, pedicelo)
        worksheet.write(row, 3, color)
        worksheet.write(row, 4, f"{str(danio).replace('.',',')}")
        row += 1
    
    workbook.close()
    ruta_directorio_1 = ruta_guardar
    ruta_directorio_2 = "image/"
    def eliminar_imagenes_en_ruta(ruta_directorio):
        archivos = os.listdir(ruta_directorio)
        for archivo in archivos:
            if archivo.endswith(".jpg") or archivo.endswith(".json"):
                ruta_archivo = os.path.join(ruta_directorio, archivo)
                os.remove(ruta_archivo)
                print(f"Se eliminó: {ruta_archivo}")
    eliminar_imagenes_en_ruta(ruta_directorio_1)

    eliminar_imagenes_en_ruta(ruta_directorio_2)


def documento(results, largo, h, ruta_guardar, name_file, image_file, tam_y, tam_x, numero_aleatorio, fecha_actual):
    # data = [("Cerezas","Calibre","Pedicelo","Danio", "Color", "Color Class", "% R. Cl.", "% R.", "% R. Ca.", "% Ca. Os.", "% N.", "danio_pudri_1", "danio_pudri_2", "danio_pudri_2_1", "danio_pudri_2_2", "danio_pudri_2_3", "danio_pudri_2_4")]
    # data = [("Cerezas","Calibre","Pedicelo", "Color", "danio_1", "danio_2", "danio_2_1", "danio_2_2", "danio_2_3", "danio_2_4")]
    data = [("Cerezas","Calibre (mm)","Pedicelo", "Color", "Daño (%)")]
    lista_color = []
    lista_calibre = []
    lista_pedicelo = []
    v = 0
    r_a = 0
    r = 0
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

    # creamos json de resultados.
    data_json = {}
    datos_array = []
    datos = {}
    solicitudImagen_id = {}
    imagenProcesada = {}
    resultadoAnalisis = []
    frutos = {}
    
    data_json['respuesta'] = 'Exito'        
    datos['solicitudImagen_id'] = 1
    datos['imagenProcesada'] = numero_aleatorio+version+fecha_actual
    
    datos_array.append(datos)
    datos_array.append(datos)

    data_json['datos'] = []

    for i in range(largo):
        color_class = results[i]["color_class"]
        porcentaje_rojo_claro = results[i]["porcentaje_rojo_claro"]
        porcentaje_rojo = results[i]["porcentaje_rojo"]
        porcentaje_rojo_caoba = results[i]["porcentaje_rojo_caoba"]
        porcentaje_caoba_oscuro = results[i]["porcentaje_caoba_oscuro"]
        porcentaje_negro = results[i]["porcentaje_negro"]
        size_x = results[i]["size_x"]
        size_y = results[i]["size_y"]
        danio = round(results[i]["danio"])        
        # danio_pudri_1 = results[i]["danio_pudri_1"]
        # danio_pudri_2 = results[i]["danio_pudri_2"]
        # danio_pudri_2_1 = results[i]["danio_pudri_2_1"]
        # danio_pudri_2_2 = results[i]["danio_pudri_2_2"]
        # danio_pudri_2_3 = results[i]["danio_pudri_2_3"]
        # danio_pudri_2_4 = results[i]["danio_pudri_2_4"]
        # fruta_sana = results[i]["fruta_sana"]
        # fruta_danada = results[i]["fruta_danada"]
        pedicelo = results[i]["pedicelo"]
        # ubicacion_i = results[i]["ubicacion_i"]
        # ubicacion_j = results[i]["ubicacion_j"]

        ## Reglas del color, se usan clase de color (distancia LAB) y rangos de % (HSV).
        ## Rojo claro
        if (color_class == 0):
            if(porcentaje_rojo_claro >= 50.1):
                color = "Rojo Claro"
                v = v + 1
            elif(porcentaje_rojo_claro < porcentaje_rojo and porcentaje_rojo >= 40.1):
                color = "Rojo"
                r_a = r_a + 1
            elif(porcentaje_rojo >= 50.1):
                color = "Rojo"
                r_a = r_a + 1
            else:
                color = "Rojo Claro"
                v = v + 1
        
        ## Rojo
        elif (color_class == 1):
            if(porcentaje_rojo_claro >= 50.1):
                color = "Rojo Claro"
                v = v + 1
            elif(porcentaje_rojo >= 50.1): 
                color = "Rojo"
                r_a = r_a + 1
            elif(porcentaje_rojo_caoba >= 50.1):
                color = "Rojo Caoba"
                r = r + 1 
            else:
                color = "Rojo"
                r_a = r_a + 1
        
        ## Rojo caoba
        elif (color_class == 2):
            if(porcentaje_rojo >= 50.1):
                color = "Rojo"
                r_a = r_a + 1
            elif(porcentaje_rojo_caoba >= 50.1): 
                color = "Rojo Caoba"
                r = r + 1  
            elif(porcentaje_caoba_oscuro >= 50.1):
                color = "Caoba Oscuro"
                o_e = o_e + 1
            else:
                color = "Rojo Caoba"
                r = r + 1

        ## Caoba oscuro
        elif (color_class == 3): ## Solo usar color class para clasificar este color.
            if(porcentaje_rojo_caoba >= 50.1):
                color = "Rojo Caoba"
                r = r + 1
            elif(porcentaje_caoba_oscuro >= 40.1):
                color = "Caoba Oscuro"
                o_e = o_e + 1    
            elif(porcentaje_negro >= 50.1):
                color = "Negro"
                N = N + 1
            elif(porcentaje_caoba_oscuro < porcentaje_negro and porcentaje_negro >= 45.1): ## Regla por evaluar
                color = "Negro"
                N = N + 1
            else:
                color = "Caoba Oscuro"
                o_e = o_e + 1  

        ## Negro
        # elif (color_class == 4 or color_class == 5):
        else:
            if(porcentaje_caoba_oscuro >= 50.1):
                color = "Caoba Oscuro"
                o_e = o_e + 1    
            elif(porcentaje_negro >= 50.1):
                color = "Negro"
                N = N + 1
            else:
                color = ''
        
        if(size_x < size_y):
            size = round(size_y,2) 
        else:
            size = round(size_x,2) 

        # if (ubicacion_i == 0): size = size*0.95
        # elif (ubicacion_i == 1): size = size*0.965
        # elif (ubicacion_i == 5): size = size*0.965
        # elif (ubicacion_i == 6): size = size*0.95

        # if (size < 22):
        #     calibre_1 = calibre_1 + 1
        # elif (size < 24 and size >= 22):
        #     calibre_2 = calibre_2 + 1
        # elif (size < 26 and size >=24):
        #     calibre_3 = calibre_3 + 1
        # elif (size < 28 and size >=26):
        #     calibre_4 = calibre_4 + 1
        # elif (size < 30 and size >=28):
        #     calibre_5 = calibre_5 + 1
        # elif (size >= 30):
        #     calibre_6 = calibre_6 + 1

        if (size < 21.99):
            calibre_1 = calibre_1 + 1
        elif (size < 23.99 and size >= 22.00):
            calibre_2 = calibre_2 + 1
        elif (size < 25.99 and size >=24.00):
            calibre_3 = calibre_3 + 1
        elif (size < 27.99 and size >=26.00):
            calibre_4 = calibre_4 + 1
        elif (size < 29.99 and size >=28.00):
            calibre_5 = calibre_5 + 1
        elif (size >= 30.00):
            calibre_6 = calibre_6 + 1

        if (pedicelo == "Si"):
            positivo = positivo + 1
        elif (pedicelo == "No"):
            negativo = negativo + 1
        
        # data.append(("Cereza " + str(i + 1), str(size), pedicelo, danio, color, color_class, porcentaje_rojo_claro, porcentaje_rojo, porcentaje_rojo_caoba, porcentaje_caoba_oscuro, porcentaje_negro, danio_pudri_1, danio_pudri_2, danio_pudri_2_1, danio_pudri_2_2, danio_pudri_2_3, danio_pudri_2_4))
        # data.append(("Cereza " + str(i + 1), str(size), pedicelo, color, danio_pudri_1, danio_pudri_2, danio_pudri_2_1, danio_pudri_2_2, danio_pudri_2_3, danio_pudri_2_4))
        data.append(("Cereza " + str(i + 1), str(size), pedicelo, color, danio))

        ##
        frutos['NumeroFruto'] = "Cereza " + str(i + 1)
        frutos['Color'] = color
        frutos['Calibre'] = f"{str(round(size,2)).replace('.',',')}" # str(round(size,2))
        frutos['Pedicelo'] = pedicelo
        frutos['Danio'] = f"{str(danio).replace('.',',')}" # danio
        # frutos['color_class'] = str(color_class)
        # frutos['porcentaje_rojo_claro'] = porcentaje_rojo_claro
        # frutos['porcentaje_rojo'] = porcentaje_rojo
        # frutos['porcentaje_rojo_caoba'] = porcentaje_rojo_caoba
        # frutos['porcentaje_caoba_oscuro'] = porcentaje_caoba_oscuro
        # frutos['porcentaje_negro'] = porcentaje_negro
        # frutos['danio_pudri_1'] = danio_pudri_1
        # frutos['danio_pudri_2'] = danio_pudri_2
        # frutos['danio_pudri_2_1'] = danio_pudri_2_1
        # frutos['danio_pudri_2_2'] = danio_pudri_2_2
        # frutos['danio_pudri_2_3'] = danio_pudri_2_3
        # frutos['danio_pudri_2_4'] = danio_pudri_2_4

        resultadoAnalisis.append(frutos)
        frutos = {}

    datos['resultadoAnalisis'] =  resultadoAnalisis
    datos_array.append(resultadoAnalisis)
    data_json['datos'].append(datos)

    ## Guardamos json
    ## GAMADIEL, ACA ESTA LO QUE BUSCAS
    archivo_json = os.path.join(ruta_guardar, "json" + numero_aleatorio + version + fecha_actual + ".json")    # print(data_json)
    with open(archivo_json, 'w') as  json_file:
        json.dump(data_json, json_file)

    lista_color.append(round(((v * 100)/largo),1))
    lista_color.append(round(((r_a * 100)/largo),1))
    lista_color.append(round(((r * 100) / largo),1))
    lista_color.append(round(((o_e * 100) / largo),1))
    lista_color.append(round(((N * 100) / largo),1))
    lista_calibre.append(round(((calibre_1 * 100) / largo),2))
    lista_calibre.append(round(((calibre_2 * 100) / largo),2))
    lista_calibre.append(round(((calibre_3 * 100) / largo),2))
    lista_calibre.append(round(((calibre_4 * 100) / largo),2))
    lista_calibre.append(round(((calibre_5 * 100) / largo),2))
    lista_calibre.append(round(((calibre_6 * 100) / largo),2))
    lista_pedicelo.append(round(((positivo * 100) / largo),2))
    lista_pedicelo.append(round(((negativo * 100) / largo),2))
  
    
    export_to_pdf(data,image_file,lista_color,lista_calibre,lista_pedicelo, name_file, h, ruta_guardar, tam_y, tam_x, numero_aleatorio, fecha_actual)
    guardar_excel(data, ruta_guardar, name_file, numero_aleatorio, fecha_actual)
    add_datos_cerezas(data_json, numero_aleatorio, fecha_actual, version)
