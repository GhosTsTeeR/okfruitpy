from prueba_algoritmos_v9_5.declarar_librerias import *

def segmentamos_color(roi_blur,roi_mask):	
	#### Antiguos rangos
	## Canales: H, S y V, rangos color rojo claro
	# lower_hsv_r_cl = np.array([0,139,36]) ## Mejorar rangos
	# upper_hsv_r_cl = np.array([120,152,37]) 

	# lower_hsv_r_cl = np.array([15,0,0]) ## Muy buenos resultados.
	# upper_hsv_r_cl = np.array([20,255,255]) 

	lower_hsv_r_cl = np.array([0,0,0]) ## Excelente resultados.
	upper_hsv_r_cl = np.array([20,255,255]) 

	### Canales: H, S y V, rangos color rojo
	# lower_hsv_r = np.array([170,125,18]) ## Buenos resultados.
	# upper_hsv_r = np.array([177,205,51]) 

	lower_hsv_r = np.array([160,125,18]) ## Excelente resultados.
	upper_hsv_r = np.array([180,205,100]) 

	## Canales: H, S y V, rangos color rojo caoba
	# lower_hsv_r_ca = np.array([123,35,11]) 
	# upper_hsv_r_ca = np.array([170,98,19]) 
	
	lower_hsv_r_ca = np.array([160,0,0])    ## Resultados regulares
	upper_hsv_r_ca = np.array([180,100,160]) 

	# ## Canales: H, S y V, rangos color caoba
	# lower_hsv_ca = np.array([115,82,17])
	# upper_hsv_ca = np.array([136,95,23])

	## Canales: H, S y V, rangos color caoba oscuro
	# lower_hsv_ca_n = np.array([110,90,16])
	# upper_hsv_ca_n = np.array([118,102,17])

	lower_hsv_ca_n = np.array([0,80,10]) ## Por mejorar
	upper_hsv_ca_n = np.array([125,125,100])

	## Canales: H, S y V, rangos color negro
	# lower_hsv_n = np.array([101,62,7])  ##
	# upper_hsv_n = np.array([106,114,18])

	# lower_hsv_n = np.array([0, 0, 9]) ## Resultados regulares.
	# upper_hsv_n = np.array([9, 92, 27])

	# lower_hsv_n = np.array([0, 0, 0]) ## Buenos resultados
	# upper_hsv_n = np.array([125, 92, 27])

	lower_hsv_n = np.array([0, 0, 0]) ##  resultados
	upper_hsv_n = np.array([125, 125, 30])

	#### HSV
	roi_hsv = cv2.cvtColor(roi_blur, cv2.COLOR_RGB2HSV)

	## rojo claro
	mask_hsv_r_cl = cv2.inRange(roi_hsv, lower_hsv_r_cl, upper_hsv_r_cl)
	roi_r_cl = cv2.bitwise_and(roi_blur, roi_blur, mask = mask_hsv_r_cl)

	## rojo
	mask_hsv_r = cv2.inRange(roi_hsv, lower_hsv_r, upper_hsv_r)
	roi_r = cv2.bitwise_and(roi_blur, roi_blur, mask = mask_hsv_r)

	## rojo caoba
	mask_hsv_r_ca = cv2.inRange(roi_hsv, lower_hsv_r_ca, upper_hsv_r_ca)
	roi_r_ca = cv2.bitwise_and(roi_blur, roi_blur, mask = mask_hsv_r_ca)
	
	## caoba oscuro
	mask_hsv_ca_n = cv2.inRange(roi_hsv, lower_hsv_ca_n, upper_hsv_ca_n)
	roi_r_ca_n = cv2.bitwise_and(roi_blur, roi_blur, mask = mask_hsv_ca_n)

	## negro
	mask_hsv_n = cv2.inRange(roi_hsv, lower_hsv_n, upper_hsv_n)
	roi_n = cv2.bitwise_and(roi_blur, roi_blur, mask = mask_hsv_n)

	return calculo_porcentaje(roi_r_cl, roi_r, roi_r_ca, roi_r_ca_n, roi_n)

##
def calculo_porcentaje(roi_r_cl, roi_r, roi_r_ca, roi_r_ca_n, roi_n):
	## sumamos pixeles rojo claro
	pixeles_r_cl = np.where(roi_r_cl[:,:,0] != pixel_negro) and np.where(roi_r_cl[:,:,1] != pixel_negro) and np.where(roi_r_cl[:,:,2] != pixel_negro)
	pixeles_r_cl_sum = (np.array(roi_r_cl[pixeles_r_cl])).sum()
		
	## sumamos pixeles rojo
	pixeles_r = np.where(roi_r[:,:,0] != pixel_negro) and np.where(roi_r[:,:,1] != pixel_negro) and np.where(roi_r[:,:,2] != pixel_negro)
	pixeles_r_sum = (np.array(roi_r[pixeles_r])).sum()

	## sumamos pixeles rojo caoba
	pixeles_r_ca = np.where(roi_r_ca[:,:,0] != pixel_negro) and np.where(roi_r_ca[:,:,1] != pixel_negro) and np.where(roi_r_ca[:,:,2] != pixel_negro)
	pixeles_r_ca_sum = (np.array(roi_r_ca[pixeles_r_ca])).sum()

	## sumamos pixeles caoba negro
	pixeles_ca_n = np.where(roi_r_ca_n[:,:,0] != pixel_negro) and np.where(roi_r_ca_n[:,:,1] != pixel_negro) and np.where(roi_r_ca_n[:,:,2] != pixel_negro)
	pixeles_ca_n_sum = (np.array(roi_r_ca_n[pixeles_ca_n])).sum()

	## sumamos pixeles negro
	pixeles_n = np.where(roi_n[:,:,0] != pixel_negro) and np.where(roi_n[:,:,1] != pixel_negro) and np.where(roi_n[:,:,2] != pixel_negro)
	pixeles_n_sum = (np.array(roi_n[pixeles_n])).sum()

	##
	if pixeles_r_cl_sum != 0 or pixeles_r_sum != 0 or pixeles_r_ca_sum != 0 or pixeles_ca_n_sum != 0 or pixeles_n_sum != 0:
		## 
		porcentaje_rojo_claro = round(float(pixeles_r_cl_sum) / float(pixeles_r_cl_sum + pixeles_r_sum + pixeles_r_ca_sum + pixeles_ca_n_sum + pixeles_n_sum) * 100,1)
		porcentaje_rojo = round(float(pixeles_r_sum) / float(pixeles_r_cl_sum + pixeles_r_sum + pixeles_r_ca_sum + pixeles_ca_n_sum + pixeles_n_sum) * 100,1)
		porcentaje_rojo_caoba = round(float(pixeles_r_ca_sum) / float(pixeles_r_cl_sum + pixeles_r_sum + pixeles_r_ca_sum + pixeles_ca_n_sum + pixeles_n_sum) * 100,1)
		porcentaje_caoba_oscuro = round(float(pixeles_ca_n_sum) / float(pixeles_r_cl_sum + pixeles_r_sum + pixeles_r_ca_sum + pixeles_ca_n_sum + pixeles_n_sum) * 100,1)
		porcentaje_negro = round(float(pixeles_n_sum) / float(pixeles_r_cl_sum + pixeles_r_sum + pixeles_r_ca_sum + pixeles_ca_n_sum + pixeles_n_sum) * 100,1)
					
	##
	else:
		porcentaje_rojo_claro = 0.0
		porcentaje_rojo = 0.0
		porcentaje_rojo_caoba = 0.0
		porcentaje_caoba_oscuro = 0.0
		porcentaje_negro = 0.0
		# clasificacion = -1 ## 'No determinada'
	
	# return [porcentaje_rojo_claro, porcentaje_rojo, porcentaje_rojo_caoba, porcentaje_caoba, porcentaje_caoba_oscuro, porcentaje_negro]
	return [porcentaje_rojo_claro, porcentaje_rojo, porcentaje_rojo_caoba, porcentaje_caoba_oscuro, porcentaje_negro]