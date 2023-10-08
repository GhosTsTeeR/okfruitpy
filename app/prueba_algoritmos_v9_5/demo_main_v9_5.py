from declarar_librerias import *
from declarar_variables import *
from declarar_rutas import *

def cargar_imagen(image_file):
    img = cv2.imread(image_file)
    return img

def preprocesar_imagen(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    ret2, th = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask1 = cv2.bitwise_and(img, img, mask=th)
    white = np.zeros_like(img)
    white = cv2.bitwise_not(white)
    mask2 = cv2.bitwise_and(white, white, mask=cv2.bitwise_not(th))
    img = mask2 + mask1
    return img

def cargar_mascaras(img_hsv):
    mask_verde = cv2.inRange(img_hsv, (35, 20, 20), (65, 255, 255))
    mask_verde_2 = cv2.inRange(img_hsv, (95, 56, 36), (98, 81, 40))
    mask_verde_3 = cv2.inRange(img_hsv, (25, 52, 72), (102, 255, 255))
    mask_verde_4 = cv2.inRange(img_hsv, (21, 149, 24), (102, 255, 255))
    mask_verde_5 = cv2.inRange(img_hsv, (4, 130, 13), (102, 255, 255))
    mask_deshidratado = cv2.inRange(img_hsv, (30, 25, 25), (95, 255, 255))
    return mask_verde, mask_verde_2, mask_verde_3, mask_verde_4, mask_verde_5, mask_deshidratado

def procesar_imagen(img, mask_verde, mask_verde_2, mask_verde_3, mask_verde_4, mask_verde_5, mask_deshidratado):
    mask = mask_verde | mask_verde_2 | mask_verde_3 | mask_verde_4 | mask_verde_5 | mask_deshidratado
    mask = cv2.dilate(mask, kernel, iterations=5)
    target = cv2.inpaint(img, mask, 2, cv2.INPAINT_NS)
    return target

# Agrega más funciones aquí para dividir las tareas en pasos más pequeños

def proceso_analisis(ruta_guardar, image_file):
    print(ruta_guardar, image_file)
  
    t_0 = time.process_time_ns()
    img = cargar_imagen(image_file)
    img_preprocesada = preprocesar_imagen(img)
    img_hsv = cv2.cvtColor(img_preprocesada, cv2.COLOR_RGB2HSV)
    mask_verde, mask_verde_2, mask_verde_3, mask_verde_4, mask_verde_5, mask_deshidratado = cargar_mascaras(img_hsv)
    target = procesar_imagen(img, mask_verde, mask_verde_2, mask_verde_3, mask_verde_4, mask_verde_5, mask_deshidratado)

    # Resto del código de procesamiento...

    print("Tiempo: " + str((time.process_time_ns() - t_0) / 1000000000) + " seg.")