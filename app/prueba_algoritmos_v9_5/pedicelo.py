from app.declarar_librerias import *
from declarar_variables import *

def deteccion_pedicelo(image, ruta_guardar, numero_fruto, version, name_file):
    img_hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    # cv2.imwrite(ruta_guardar+"img_hsv_"+str(numero_fruto) + version + name_file, img_hsv)

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
    # mask_deshidratado = cv2.inRange(img_hsv,(30,25,25),(95,255,255)) ## Buenos resultados.
    # cv2.imwrite(ruta_guardar+"mask_deshidratado" + version + name_file, mask_deshidratado)

    # mask_deshidratado_2 = cv2.inRange(img_hsv,(109,57,17),(160,92,33)) ## Buenos resultados.
    # cv2.imwrite(ruta_guardar+"mask_deshidratado_2" + version + name_file, mask_deshidratado_2)

    # mask_deshidratado_3 = cv2.inRange(img_hsv,(100,50,10),(160,95,35)) ## Buenos resultados.
    # cv2.imwrite(ruta_guardar+"mask_deshidratado_3" + version + name_file, mask_deshidratado_3)

    # mask_deshidratado_4 = cv2.inRange(img_hsv,(110,58,17),(160,120,33)) ## Buenos resultados.
    # cv2.imwrite(ruta_guardar+"mask_deshidratado_4" + version + name_file, mask_deshidratado_4)

    # mask_bandeja = cv2.inRange(img_hsv,(103,49,50),(107,67,76)) ## Buenos resultados.
    # cv2.imwrite(ruta_guardar+"mask_bandeja" + version + name_file, mask_bandeja)

    ## 
    # mask = mask_verde | mask_verde_2 | mask_deshidratado | mask_deshidratado_2 | mask_deshidratado_3 | mask_deshidratado_4 | mask_bandeja
    mask = mask_verde | mask_verde_2 | mask_verde_3 | mask_verde_4 | mask_verde_5
    # cv2.imwrite(ruta_guardar+"mask_pedicelo_"+str(numero_fruto) + version + name_file, mask)
    
    ## 
    porcentaje_pedicelo = porcentaje(image,mask)
    area_pedicelo, perimetro_pedicelo = area_perimeter(mask,image)

    # print(porcentaje_pedicelo < 0.33 or area_pedicelo < 33.00)
    if(porcentaje_pedicelo < 0.33 and area_pedicelo < 33.00):
        tiene_pedicelo = "No"
    elif(porcentaje_pedicelo >= 0.33): # or area_pedicelo >= 33):
        tiene_pedicelo = "Si"
    elif(area_pedicelo >= 33.00):
        tiene_pedicelo = "Si"
    elif(perimetro_pedicelo >= 2.00):
        tiene_pedicelo = "Si"
    else:
        tiene_pedicelo = "No"
    
    # print(numero_fruto, porcentaje_pedicelo, area_pedicelo, perimetro_pedicelo, tiene_pedicelo)
    return tiene_pedicelo


def porcentaje(image,mask):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cantidad_total_imagen = cv2.countNonZero(image_gray)
    cantidad_pixeles_no_cero = cv2.countNonZero(mask)    

    if(cantidad_pixeles_no_cero == 0):
        porcentaje = 0.0
    else:
        porcentaje = (cantidad_pixeles_no_cero / cantidad_total_imagen) * 100

    return round(porcentaje,2)


def area_perimeter(mask,image):
    _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if(len(contours) == 0):
        area = 0
        perimetro = 0
    else:
        cnt = contours[0]    
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)  # Perimeter of first contour 
    return round(area,2), round(perimetro,2) #, area_pc


# def area(mask, image):
#     # extract the contours
#     _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # blank_image = np.zeros((image.shape),np.uint8)
#     image_area = np.prod(image.shape)

#     # iterate through the contours detected from right top corner
#     # for i, c in enumerate(contours[::-1]):

#     # turn blank_image black
#     # blank_image *= 0

#     # draw filled contour
#     # cv2.drawContours(blank_image, [c], 0, (255), thickness=cv2.FILLED)

#     contour_area = cv2.contourArea(contours[0])
    
#     # percentage of area contour
#     contour_area_pc = np.true_divide(int(contour_area),image_area)*100 if int(contour_area) > 1  else 0 
#     # text = ' '.join(['Contour:',str(i),'Area:',str(round(contour_area,2)),'Percentage Area:',str(round(contour_area_pc,2))])
#     # cv2.putText(blank_image,text,(10,60), cv2.FONT_HERSHEY_SIMPLEX, 1,(255),2,cv2.LINE_AA)
    
#     # plt.imshow(blank_image, cmap = 'gray', interpolation = 'bicubic')
#     # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#     # plt.show()
#     # print(contour_area, contour_area_pc)
#     return contour_area, contour_area_pc