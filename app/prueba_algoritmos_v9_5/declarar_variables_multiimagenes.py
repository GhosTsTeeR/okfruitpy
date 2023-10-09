from prueba_algoritmos_v9_5.declarar_librerias import *

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

# # defecto: 300, 300
# # android (silvia): 320, 320
# # iphone 11: 324, 324

constante_mm = 26.4583333337192
dpi_x, dpi_y = 210, 210 # 220, 220 # 300, 300 # 425, 425

version = "_v9_5_"
pixel_negro = 0
pixel_uno = 1

img = None

marco = 27 # 31

Categories = []
Categories.append("Fruta sana")
Categories.append("Fruta dañada")