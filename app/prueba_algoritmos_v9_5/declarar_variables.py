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

kernel = np.ones((5,5), np.uint8)

# defecto: 300, 300
# android (silvia): 320, 320
# iphone 11: 324, 324

## buenos resultados con 200
## buenos resultados con 240
## excelentes resultados con 290

pixel_inch = 210 # 300 #290 # 240 # 200
cantidad_iteraciones = 1 # 4 # 3 # 2 # 1
constante_mm = 26.4583333337192 # 25.4
dpi_x, dpi_y = pixel_inch, pixel_inch # 300, 300 # 425, 425

version = "_v9_5_"
pixel_negro = 0
pixel_uno = 1

marco = 27 # 31

Categories = []
Categories.append("Fruta sana")
Categories.append("Fruta dañada")