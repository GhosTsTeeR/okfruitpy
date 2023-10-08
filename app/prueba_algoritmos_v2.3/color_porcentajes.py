from declarar_librerias import *

def segmentamos_color(roi_blur,roi_mask):	
    ## Canales: H, S y V, rangos color a claro
	lower_hsv_a_rojizo = np.array([128, 55, 21]) ##
	upper_hsv_a_rojizo = np.array([136, 71, 79]) 

	## Canales: H, S y V, rangos color a claro
	lower_hsv_a_claro = np.array([106, 65, 12]) ##
	upper_hsv_a_claro = np.array([110, 143, 86]) 

	### Canales: H, S y V, rangos color a optimo
	lower_hsv_a_optimo = np.array([111, 107, 14])
	upper_hsv_a_optimo = np.array([113, 117, 48]) 

	## Canales: H, S y V, rangos color negro
	# lower_hsv_negro = np.array([0, 0, 0]) ## 
	# upper_hsv_negro = np.array([115, 56, 11])

	lower_hsv_negro = np.array([110, 55, 4]) ## 
	upper_hsv_negro = np.array([115, 136, 11])

	#### HSV
	roi_hsv = cv2.cvtColor(roi_blur, cv2.COLOR_RGB2HSV)

    ## a rojizo
	mask_hsv_a_rojizo = cv2.inRange(roi_hsv, lower_hsv_a_rojizo, upper_hsv_a_rojizo)
	roi_a_rojizo = cv2.bitwise_and(roi_blur, roi_blur, mask = mask_hsv_a_rojizo)

	## a claro
	mask_hsv_a_claro = cv2.inRange(roi_hsv, lower_hsv_a_claro, upper_hsv_a_claro)
	roi_a_claro = cv2.bitwise_and(roi_blur, roi_blur, mask = mask_hsv_a_claro)

	## a optimo
	mask_hsv_a_optimo = cv2.inRange(roi_hsv, lower_hsv_a_optimo, upper_hsv_a_optimo)
	roi_a_optimo = cv2.bitwise_and(roi_blur, roi_blur, mask = mask_hsv_a_optimo)

	## negro
	mask_hsv_negro = cv2.inRange(roi_hsv, lower_hsv_negro, upper_hsv_negro)
	roi_negro = cv2.bitwise_and(roi_blur, roi_blur, mask = mask_hsv_negro)

	return calculo_porcentaje(roi_a_rojizo, roi_a_claro, roi_a_optimo, roi_negro)

##
def calculo_porcentaje(roi_a_rojizo, roi_a_claro, roi_a_optimo, roi_negro):
	## sumamos pixeles rojo claro
	pixeles_a_rojizo = np.where(roi_a_rojizo[:,:,0] != pixel_negro) and np.where(roi_a_rojizo[:,:,1] != pixel_negro) and np.where(roi_a_rojizo[:,:,2] != pixel_negro)
	pixeles_a_rojizo_sum = (np.array(roi_a_rojizo[pixeles_a_rojizo])).sum()

    ## sumamos pixeles rojo claro
	pixeles_a_claro = np.where(roi_a_claro[:,:,0] != pixel_negro) and np.where(roi_a_claro[:,:,1] != pixel_negro) and np.where(roi_a_claro[:,:,2] != pixel_negro)
	pixeles_a_claro_sum = (np.array(roi_a_claro[pixeles_a_claro])).sum()
		
	## sumamos pixeles rojo
	pixeles_a_optimo = np.where(roi_a_optimo[:,:,0] != pixel_negro) and np.where(roi_a_optimo[:,:,1] != pixel_negro) and np.where(roi_a_optimo[:,:,2] != pixel_negro)
	pixeles_a_optimo_sum = (np.array(roi_a_optimo[pixeles_a_optimo])).sum()

	## sumamos pixeles negro
	pixeles_negro = np.where(roi_negro[:,:,0] != pixel_negro) and np.where(roi_negro[:,:,1] != pixel_negro) and np.where(roi_negro[:,:,2] != pixel_negro)
	pixeles_negro_sum = (np.array(roi_negro[pixeles_negro])).sum()

	##
	if pixeles_a_rojizo_sum != 0 or pixeles_a_claro_sum != 0 or pixeles_a_optimo_sum != 0 or pixeles_negro_sum != 0:
		## 
		porcentaje_a_rojizo = round(float(pixeles_a_rojizo_sum) / float(pixeles_a_rojizo_sum + pixeles_a_claro_sum + pixeles_a_optimo_sum + pixeles_negro_sum) * 100,1)
		porcentaje_a_claro = round(float(pixeles_a_claro_sum) / float(pixeles_a_rojizo_sum + pixeles_a_claro_sum + pixeles_a_optimo_sum + pixeles_negro_sum) * 100,1)
		porcentaje_a_optimo = round(float(pixeles_a_optimo_sum) / float(pixeles_a_rojizo_sum + pixeles_a_claro_sum + pixeles_a_optimo_sum + pixeles_negro_sum) * 100,1)
		porcentaje_negro = round(float(pixeles_negro_sum) / float(pixeles_a_rojizo_sum + pixeles_a_claro_sum + pixeles_a_optimo_sum + pixeles_negro_sum) * 100,1)
					
	##
	else:
		porcentaje_a_rojizo = 0.0
		porcentaje_a_claro = 0.0
		porcentaje_a_optimo = 0.0
		porcentaje_negro = 0.0

	# return [porcentaje_rojo_claro, porcentaje_rojo, porcentaje_rojo_caoba, porcentaje_caoba, porcentaje_caoba_oscuro, porcentaje_negro]
	return [porcentaje_a_rojizo, porcentaje_a_claro, porcentaje_a_optimo, porcentaje_negro]