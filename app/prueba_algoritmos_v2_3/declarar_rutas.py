from prueba_algoritmos_v2_3.declarar_librerias import *

## Imagenes testing
# image_file = os.path.join(r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC - OKFruitApp/imágenes/testing/2.jpg')
# image_file = os.path.join(r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC - OKFruitApp/imágenes/testing/3.jpg')
# image_file = os.path.join(r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC - OKFruitApp/imágenes/testing/4.jpg')

## Imagenes de dataset
# image_file = os.path.join(r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC - OKFruitApp/imágenes/arandanos brightwell/Fabi arriba 18MP.jpg')
# image_file = os.path.join(r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC - OKFruitApp/imágenes/arandanos brightwell/Fabi arriba 24MP.jpg')

# image_file = os.path.join(r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC - OKFruitApp/imágenes/03-11-2022 (Arándano)/Arándano 1.jpg')
# image_file = os.path.join(r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC - OKFruitApp/imágenes/24-11-2022 (Cerezas)/IMG_20221117_111635_008.jpg')
# image_file = os.path.join(r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC - OKFruitApp/imágenes/24-11-2022 (Cerezas)/IMG_20221117_111635_008_2.jpg')

image_file = r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC - OKFruitApp/imágenes/25-11-2022 (Arándanos)/IMG_20221125_171401_661.jpg'


## Datasets 20-05-2022
# image_file = os.path.join(r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC - OKFruitApp/imágenes/20-05-2022 (Cerezas y Arándanos)/Arándano-20220520T160037Z-001/20220110_120628.jpg')


# image_file_base64 = os.path.join(r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC - OKFruitApp/imágenes/testing/2.bin')

# file = open(image_file_base64, 'rb')
# byte = file.read()
# file.close()
  
# decodeit = open('2.jpeg', 'wb')
# decodeit.write(base64.b64decode((byte)))
# decodeit.close()

# name_file = 'temp_img.jpeg'
name_file = image_file.split("/")[-1]
name_folder = image_file.split("/")[-2]

# decodeit = open(name_file, 'wb')
# decodeit.write(base64.b64decode((byte)))
# decodeit.close()

# image = cv2.imread(name_file)

# cv2.imshow('image',cv2.imread(name_file))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## Ruta para guardar resultados
name_file = name_file.replace(" ","_")
ruta_guardar = os.path.join(r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC - OKFruitApp/resultados/' + name_file + version +'/')

if not os.path.exists(ruta_guardar):
    os.makedirs(ruta_guardar)