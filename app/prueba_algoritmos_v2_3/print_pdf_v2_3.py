from prueba_algoritmos_v2_3.declarar_librerias import *
from prueba_algoritmos_v2_3.declarar_variables import *
from __init__ import add_datos_arandanos
def grouper(iterable,n):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args)

def export_to_pdf(data, image_file, lista_color, lista_calibre, ruta_guardar, name_file, h, numero_aleatorio, fecha_actual):
    c = canvas.Canvas(ruta_guardar+"pdf"+numero_aleatorio+version+fecha_actual+".pdf", pagesize=A4)
    c.drawString(400,800, "Reporte Arandano")
    c.drawString(120,750, "Resumen de color")
    for i in range(0,4):
        # if (lista_color[0] >= 0):
        #    c.drawString(100,700, "Porcentaje de Arándanos de Color Verde : "+ str(lista_color[0])+" %")
        # if (lista_color[1] >= 0):
        #    c.drawString(100,680, "Porcentaje de Arándanos de Color Roja Punta Amarillo : "+ str(lista_color[1])+" %")
        # if (lista_color[2] >= 0):
        #    c.drawString(100,660, "Porcentaje de Arándanos de Color Rojo : "+ str(lista_color[2])+" %")
        # if (lista_color[3] >= 0):
        #    c.drawString(100,640, "Porcentaje de Arándanos de Color Purpura : "+ str(lista_color[3])+" %")
        # if (lista_color[4] >= 0):
        #    c.drawString(100,620, "Porcentaje de Arándanos de Color Optimo Exportación : "+ str(lista_color[4])+" %")
        if (lista_color[0] >= 0):
           c.drawString(100,700, "Porcentaje de Arándanos de Color A. Rojizo : "+ str(lista_color[0])+" %")
        if (lista_color[1] >= 0):
           c.drawString(100,680, "Porcentaje de Arándanos de Color A. Claro : "+ str(lista_color[1])+" %")
        if (lista_color[2] >= 0):
           c.drawString(100,660, "Porcentaje de Arándanos de Color A. Óptimo. : "+ str(lista_color[2])+" %")
        if (lista_color[3] >= 0):
           c.drawString(100,640, "Porcentaje de Arándanos de Color Negro : "+ str(lista_color[3])+" %")
    
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
    x_list = [ x + x_offset for x in [0, 100, 220, 320, 420, 520]]
    y_list = [ b - y_offset - i*padding for i in range(max_rows_per_page + 1)]

    for rows in grouper(data, max_rows_per_page):
        rows = tuple(filter(bool, rows))
        c.grid(x_list,y_list[:len(rows) + 1])
        for y, row in zip(y_list[:-1], rows):
            for x, cell in zip(x_list, row):
                c.drawString(x + 2, y - padding + 3, str(cell))
        c.showPage()
    # print(name_file)
    c.drawImage(ruta_guardar + 'img'+numero_aleatorio+version+fecha_actual+".jpg", 80, h - 50, width = 490, height = 700)
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
    datos['imagenProcesada'] = 'temp.jpeg'
    
    datos_array.append(datos)
    datos_array.append(datos)

    data_json['datos'] = []

    frutos['NumeroFruto'] = ""
    frutos['Color'] = ""
    frutos['Calibre'] = ""
    
    # frutos['Pedicelo'] = ""
    frutos['Danio'] = ""
    frutos['Manipulacion'] = ""
    resultadoAnalisis.append(frutos)
    frutos = {}

    datos['resultadoAnalisis'] = resultadoAnalisis
    datos_array.append(resultadoAnalisis)
    data_json['datos'].append(datos)

    ## Guardamos json
    ## GAMADIEL, ACA ESTA LO QUE BUSCAS
    archivo_json = os.path.join(ruta_guardar+"json" + version +".json")
    with open(archivo_json, 'w') as json_file:
        json.dump(data_json, json_file)

## 
def guardar_excel(data,ruta_guardar, name_file, numero_aleatorio, fecha_actual):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(ruta_guardar + "excel"+numero_aleatorio+version+fecha_actual+".xlsx")
    worksheet = workbook.add_worksheet()
    
    # Start from the first cell. Rows and columns are zero indexed.
    row = 0

    # Iterate over the data and write it out row by row.
    for fruto, color, tamanio, danio, manipulacion in (data):
        worksheet.write(row, 0, fruto)
        worksheet.write(row, 1, color)
        worksheet.write(row, 2, f"{str(tamanio).replace('.',',')}")
        worksheet.write(row, 3, f"{str(danio).replace('.',',')}")
        worksheet.write(row, 4, f"{str(manipulacion).replace('.',',')}")
        # worksheet.write(row, 5, porcentaje_a_rojizo)
        # worksheet.write(row, 6, porcentaje_a_claro)
        # worksheet.write(row, 7, porcentaje_a_optimo)
        # worksheet.write(row, 8, porcentaje_negro)
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


def documento(results, largo, image_file, ruta_guardar, name_file, h, numero_aleatorio, fecha_actual):
    data = [("Arándano","Color","Calibre (mm)","Daño (%)","Manipulación (%)")]

    lista_color = []
    lista_calibre = []
    a_ro = 0
    a_cl = 0
    a_o = 0
    ne = 0
    calibre_1 = 0
    calibre_2 = 0
    calibre_3 = 0
    calibre_4 = 0
    calibre_5 = 0
    calibre_6 = 0
    calibre_7 = 0
    indice = 0

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
    datos['imagenProcesada'] = 'temp.jpeg'
    
    datos_array.append(datos)

    data_json['datos'] = []

    for i in range(largo):
        # print(results[i])
        color_class = results[i]["color_class"]
        porcentaje_a_claro = results[i]["porcentaje_a_claro"]
        danio = results[i]["danio"]
        manipulacion = results[i]["manipulacion"]
        size_x = results[i]["size_x"]
        size_y = results[i]["size_y"]
        ubicacion_i = results[i]["ubicacion_i"]
        ubicacion_j = results[i]["ubicacion_j"]
        
        ## Color A Rojizo.
        if (color_class == 0):
            color = "A. Rojizo"
            a_ro = a_ro + 1
        ## Color A. Claro.
        elif (color_class == 1):
            color = "A. Claro"
            a_cl = a_cl + 1
        ## Color A. Óptimo o púrpura.
        if (color_class == 2):
            if porcentaje_a_claro > 89.0:
                color = "A. Claro"
                a_cl = a_cl + 1
            elif porcentaje_a_claro <= 89.0:
                color = "A. Óptimo"
                a_o = a_o + 1
            else:
                color = "A. Óptimo"
                a_o = a_o + 1
        ## Color Negro
        if (color_class == 3):
            color = "Negro"
            ne = ne + 1


        if(size_x < size_y):
            size = round(size_y,2)
        else:
            size = round(size_x,2)

        ## reajusta por ubicación para eliminar calibres que dan mas alto.
        if (ubicacion_i == 0): size = size*0.95
        if (ubicacion_i == 1): size = size*0.965
        if (ubicacion_i == 5): size = size*0.965
        if (ubicacion_i == 6): size = size*0.95


        if (size < 11.99):
            calibre_1 = calibre_1 + 1
        elif (size < 13.99 and size >= 12.00):
            calibre_2 = calibre_2 + 1
        elif (size < 15.99 and size >=14.00):
            calibre_3 = calibre_3 + 1
        elif (size < 17.99 and size >=16.00):
            calibre_4 = calibre_4 + 1
        elif (size < 19.99 and size >=18.00):
            calibre_5 = calibre_5 + 1
        elif (size < 21.99 and size >=20.00):
            calibre_6 = calibre_6 + 1
        elif (size >= 22.00):
            calibre_7 = calibre_7 + 1

        size = round(size,2)

        data.append((f"Arandano {i+1}", color, f" {size}", f"{danio}", f"{manipulacion}"))
    
        ##
        frutos['NumeroFruto'] = "Arándano " + str(i + 1)
        frutos['Color'] = color
        frutos['Calibre'] = f"{str(round(size,2)).replace('.',',')}"
        frutos['Danio'] = f"{str(danio).replace('.',',')}" # str(danio)
        frutos['Manipulacion'] = f"{str(manipulacion).replace('.',',')}" # str(manipulacion)

        resultadoAnalisis.append(frutos)
        frutos = {}

    datos['resultadoAnalisis'] =  resultadoAnalisis
    datos_array.append(resultadoAnalisis)
    data_json['datos'].append(datos)

    ## Guardamos json
    archivo_json = os.path.join(ruta_guardar+"json"+numero_aleatorio+version+fecha_actual+".json")
    with open(archivo_json, 'w') as  json_file:
        json.dump(data_json, json_file)

    lista_color.append(round(((a_ro * 100) / largo),2))
    lista_color.append(round(((a_cl * 100) / largo),2))
    lista_color.append(round(((a_o * 100) / largo),2))
    lista_color.append(round(((ne * 100) / largo),2))
    lista_calibre.append(round(((calibre_1 * 100) / largo),2))
    lista_calibre.append(round(((calibre_2 * 100) / largo),2))
    lista_calibre.append(round(((calibre_3 * 100) / largo),2))
    lista_calibre.append(round(((calibre_4 * 100) / largo),2))
    lista_calibre.append(round(((calibre_5 * 100) / largo),2))
    lista_calibre.append(round(((calibre_6 * 100) / largo),2))
    lista_calibre.append(round(((calibre_7 * 100) / largo),2))

    add_datos_arandanos(data_json, numero_aleatorio, fecha_actual, version)
    export_to_pdf(data,image_file,lista_color,lista_calibre, ruta_guardar, name_file, h, numero_aleatorio, fecha_actual)
    guardar_excel(data,ruta_guardar, name_file, numero_aleatorio, fecha_actual)