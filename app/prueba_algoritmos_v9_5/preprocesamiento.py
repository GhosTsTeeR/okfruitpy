from app.declarar_librerias import *

#### Probar con disminuir las dimensiones de las imagenes hasta obtener un resultado aceptable.
#### Revisar dimensiones de imagenes testing con las nuevas imagenes.
# limitamos la fotografia a un tamaño y la reescalamos
def limit_resolution(image):
    ## valores de limit_h = 1032 y limit_w = 774 que pertenecen a imagenes test.
    # limit_h = 1032 
    # limit_w = 774 
    # img_h = image.shape[0]
    # img_w = image.shape[1]
    # if (img_h > img_w):
    #     scale = limit_h / img_h
    # else:
    #     scale = limit_w / img_w
    # image = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_LINEAR)
    # return image

    limit_w = 774  # Max width for the image
    limit_h = 1032    # Max height for the image
    ratio = 0  # Used for aspect ratio
    img_w = image.shape[1]   # Current image width
    img_h = image.shape[0]  # Current image height

    if(img_w > limit_w and img_h > limit_h):
        if (img_h > img_w):
            scale = limit_h / img_h
        else:
            scale = limit_w / img_w
        
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_LINEAR)
        
    return image