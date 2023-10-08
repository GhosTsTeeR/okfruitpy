from prueba_algoritmos_v2_3.declarar_librerias import *

## Calculamos mascara usando intervalo de colores.
def recognizer(image,low,up):
	lower = np.array(low, dtype = "uint8") # Se define el limite inferior en RGB
	upper = np.array(up, dtype = "uint8")  # Se define el limite superior en RGB
	mask = cv2.inRange(image, lower, upper) # Se crea la mascara
	return mask


def calculate_pixeles_sum(image, ruta_guardar, numero_fruto, name_file):
	image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

	## Toda la fruta
	mask = recognizer(image_hsv,[0,0,0],[255,255,255])
	roi = cv2.bitwise_and(image_hsv, image_hsv, mask = None)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_roi_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi, cv2.COLOR_HSV2BGR))

	pixeles = np.where(roi[:,:,0] != pixel_negro) and np.where(roi[:,:,1] != pixel_negro) and np.where(roi[:,:,2] != pixel_negro)
	pixeles_sum = (np.array(roi[pixeles])).sum()
	return pixeles_sum


# def glcm(image, mask, ruta_guardar, numero_fruto, name_file, algorithm):
# 	##
# 	from skimage.feature import greycomatrix, greycoprops
# 	from skimage import io, color, img_as_ubyte
# 	import matplotlib
# 	import matplotlib.pyplot as plt

# 	image_input = cv2.bitwise_and(image, image, mask=mask)
# 	cv2.imwrite(ruta_guardar + algorithm + "_image_input_" + str(numero_fruto) + version + name_file, image_input)

# 	_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 	xs = []
# 	ys = []

# 	dissimilarity = []
# 	correlation = []
# 	contrast = []
# 	homogeneity = []
# 	energy = []
# 	asm = []
# 	width, height, _ = image.shape
# 	for index, cnt in enumerate(contours):
# 		x,y,w,h = cv2.boundingRect(cnt)
# 		# print(x,y,w,h,w*h,y+h,x+w)
# 		# print(w*h,w,width,h,height)
		
# 		if (w*h > 25 and (w < width and h < height)):
# 			image_roi = image[y:y+h, x:x+w]
# 			# cv2.imwrite(ruta_guardar + algorithm + "_" + str(index) + "_image_roi_" + str(numero_fruto) + version + name_file, image_roi)

# 			## BUSCAR COMO NO CONCIDERAR PIXELES NEGROS O BLANCOS (FILTRAR).
# 			distances = [1]
# 			angles = [0]
# 			levels = 256
# 			for patch in (image_roi):
# 				glcm = greycomatrix(patch, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
# 				xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
# 				ys.append(greycoprops(glcm, 'correlation')[0, 0])

# 				dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0, 0])
# 				correlation.append(greycoprops(glcm, 'correlation')[0, 0])
# 				contrast.append(greycoprops(glcm, 'contrast')[0, 0])
# 				homogeneity.append(greycoprops(glcm, 'homogeneity')[0, 0])
# 				energy.append(greycoprops(glcm, 'energy')[0, 0])
# 				asm.append(greycoprops(glcm, 'ASM')[0, 0])

# 	## DAR INTERPRETACION A ESTA GRAFICA, TAMBIEN COMO REPRESENTAR LOS XS E YS APARTE PARA QUE NO SE SOLAPEN
# 	if xs and ys:
# 		fig = plt.figure()  # create a figure object
# 		ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
# 		ax.plot(xs[:len(image_roi)], ys[:len(image_roi)], 'x', label='Fruta')
# 		ax.plot(xs[len(image_roi):], ys[len(image_roi):], '+', label='Daño')
# 		ax.set_xlabel('GLCM Dissimilarity')
# 		ax.set_ylabel('GLCM Correlation')
# 		ax.legend()
# 		fig.savefig(ruta_guardar + algorithm + "_GLMC_" + str(numero_fruto) + version + name_file, bbox_inches='tight')
# 		plt.close('all')

# 		# print("Algoritmo: ", algorithm, "Nro. Fruto: ", numero_fruto)
# 		# print("Fruta: ", xs[:len(image_roi)], ys[:len(image_roi)])
# 		# print("Daño: ", xs[len(image_roi):], ys[len(image_roi):])
	
# 	# with open(ruta_guardar + algorithm + '_dissimilarity_' + str(numero_fruto) + version + name_file + '.txt', 'w') as f:
# 	# 	f.write(str(dissimilarity))

# 	# with open(ruta_guardar + algorithm + '_correlation_' + str(numero_fruto) + version + name_file + '.txt', 'w') as f:
# 	# 	f.write(str(correlation))

# 	# with open(ruta_guardar + algorithm + '_contrast_' + str(numero_fruto) + version + name_file + '.txt', 'w') as f:
# 	# 	f.write(str(contrast))

# 	# with open(ruta_guardar + algorithm + '_homogeneity_' + str(numero_fruto) + version + name_file + '.txt', 'w') as f:
# 	# 	f.write(str(homogeneity))

# 	# with open(ruta_guardar + algorithm + '_energy_' + str(numero_fruto) + version + name_file + '.txt', 'w') as f:
# 	# 	f.write(str(energy))

# 	# with open(ruta_guardar + algorithm + '_asm_' + str(numero_fruto) + version + name_file + '.txt', 'w') as f:
# 	# 	f.write(str(asm))
	

# # def danio_pudricion_8(image,ruta_guardar,numero_fruto,name_file):
# # 	# from skimage.transform import resize
# # 	# url=input('Enter URL of Image :')
# # 	# img=imread(url)
# # 	# plt.imshow(img)
# # 	# plt.show()

# # 	from declarar_librerias import resize

# # 	ruta = r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC/resultados/No Borrar/model_91_6_accurate.p'

# # 	model = pd.read_pickle(ruta)

# # 	img_resize = resize(image,(150,150,3))
# # 	l = [img_resize.flatten()]
# # 	probability = model.predict_proba(l)
	
# # 	fruta_sana = 0
# # 	fruta_danada = 0
# # 	for ind, val in enumerate(Categories):
# # 	# 	print("Numero fruto: ", numero_fruto, " ", f'{val} = {probability[0][ind]*100}%', " The predicted image is : ", Categories[model.predict(l)[0]])
# # 	# 	print("Numero fruto: ", numero_fruto)
# # 	# 	print(f'{val} = {probability[0][ind]*100}%')
# # 	# 	print("The predicted image is : "+Categories[model.predict(l)[0]])
# # 		if (Categories[model.predict(l)[0]] == "Fruta sana"):
# # 			fruta_sana = round(probability[0][ind]*100,2)
# # 		else:
# # 			fruta_danada = round(probability[0][ind]*100,2)

# # 	return fruta_sana, fruta_danada


# # def danio_pudricion_7(image,ruta_guardar,numero_fruto,name_file):
# # 	# #Inicialice el elemento de corte, el tamaño promedio de los superpíxeles es 20 (el valor predeterminado es 10) y el factor de suavizado es 20
# # 	# slic = cv2.ximgproc.createSuperpixelSLIC(image,region_size=20,ruler = 20.0) 
# # 	# slic.iterate(200)	 # Número de iteraciones, cuanto mayor, mejor
# # 	# mask_slic = slic.getLabelContourMask() #Get Mask, Super pixel edge Mask == 1
# # 	# # label_slic = slic.getLabels()		# Obtener etiquetas de superpíxeles
# # 	# # number_slic = slic.getNumberOfSuperpixels()  # Obtenga el número de superpíxeles
# # 	# mask_inv_slic = cv2.bitwise_not(mask_slic)  
# # 	# img_slic = cv2.bitwise_and(image,image,mask =  mask_inv_slic)
	
# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_7_slic_" + str(numero_fruto) + version + name_file, img_slic)

# # 	# #Inicialice el elemento de semillas, preste atención al orden del largo y ancho de la imagen
# # 	# seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1],image.shape[0],image.shape[2],2000,15,3,5,True)
# # 	# seeds.iterate(image,10)  # El tamaño de la imagen de entrada debe ser el mismo que la forma inicial, el número de iteraciones es 10
# # 	# mask_seeds = seeds.getLabelContourMask()
# # 	# # label_seeds = seeds.getLabels()
# # 	# # number_seeds = seeds.getNumberOfSuperpixels()
# # 	# mask_inv_seeds = cv2.bitwise_not(mask_seeds)
# # 	# img_seeds = cv2.bitwise_and(image,image,mask =  mask_inv_seeds)

# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_7_seeds_" + str(numero_fruto) + version + name_file, img_seeds)

# # 	lsc = cv2.ximgproc.createSuperpixelLSC(image)
# # 	lsc.iterate(20)
# # 	# print(lsc)
# # 	mask_lsc = lsc.getLabelContourMask()
# # 	# print(mask_lsc)
# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_7_mask_lsc_" + str(numero_fruto) + version + name_file, mask_lsc)

# # 	# label_lsc = lsc.getLabels()
# # 	# print(label_lsc)
# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_7_label_lsc_" + str(numero_fruto) + version + name_file, label_lsc)

# # 	# number_lsc = lsc.getNumberOfSuperpixels()
# # 	# print(number_lsc)

# # 	mask_inv_lsc = cv2.bitwise_not(mask_lsc)
# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_7_mask_inv_lsc_" + str(numero_fruto) + version + name_file, mask_inv_lsc)

# # 	# img_lsc = cv2.bitwise_and(image,image,mask = mask_inv_lsc)
# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_7_lsc_" + str(numero_fruto) + version + name_file, img_lsc)

# # 	# print(contrast, dissimilarity, homogeneity, energy, correlation, ASM)

# # 	glcm(image, mask_inv_lsc, ruta_guardar, numero_fruto, name_file, "danio_pudricion_7")


# # def danio_pudricion_6(image,ruta_guardar,numero_fruto,name_file):
# # 	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 	from skimage.feature import local_binary_pattern
# # 	import matplotlib.pyplot as plt

# # 	radius = 1
# # 	numPoints = 8 * radius 
	
# # 	lbp = local_binary_pattern(image_gray, numPoints, radius)
# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_6_" + str(numero_fruto) + version + name_file, lbp)
	
# # 	_, binary = cv2.threshold(lbp.astype('uint8'), 126, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_6_binary_" + str(numero_fruto) + version + name_file, binary)

# # 	glcm(image, binary, ruta_guardar, numero_fruto, name_file, "danio_pudricion_6")

# # 	# # hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
# # 	# n_bins = int(lbp.max() + 1)
# # 	# hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
# # 	# plt.plot(hist)
# # 	# plt.savefig(ruta_guardar+"danio_pudricion_6_hist_" + str(numero_fruto) + version + name_file)


# # def danio_pudricion_5(image,ruta_guardar,numero_fruto,name_file):
# # 	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 	from skimage.filters import threshold_otsu
# # 	from skimage.filters.rank import entropy
# # 	from skimage.morphology import disk

# # 	entropy_image = entropy(image_gray, disk(5))
# # 	thresh = threshold_otsu(entropy_image)
# # 	# thresh, _ = cv2.threshold(entropy_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# # 	# binary = entropy_image <= thresh
# # 	_, binary = cv2.threshold(entropy_image.astype('uint8'), thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_5_binary_" + str(numero_fruto) + version + name_file, binary)

# # 	# binary_inv = cv2.bitwise_not(binary)
# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_5_binary_inv_" + str(numero_fruto) + version + name_file, binary_inv*255)

# # 	glcm(image, binary, ruta_guardar, numero_fruto, name_file, "danio_pudricion_5")


# # # def danio_pudricion_4(image,ruta_guardar,numero_fruto,name_file):
# # 	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 	## Resultados regulares
# # 	# ksize = 5
# # 	# theta = np.pi/3

# # 	## Buenos resultados
# # 	# ksize = 3
# # 	# theta = np.pi/2

# # 	## Buenos Resultados
# # 	# ksize = 3
# # 	# theta = np.pi/4

# # 	## Buenos Resultados
# # 	# ksize = 3
# # 	# theta = np.pi/6

# # 	## Buenos Resultados
# # 	# ksize = 3
# # 	# theta = np.pi/8

# # 	ksize = 3
# # 	theta = 0
# # 	# sigma = 5.0
# # 	width, height = image_gray.shape
# # 	sigma = int(math.sqrt(max(width,height)))
# # 	# print(sigma)
# # 	# lambd = 5.0
# # 	lambd = max(width,height)
# # 	gamma = 1.0
# # 	psi = 0

# # 	# filters = []
# # 	# for theta in np.arange(0, np.pi, np.pi/16):
# # 	kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
# # 	# kernel /= 1.5*kernel.sum()
# # 	# filters.append(kernel)

# # 	# image_total = np.zeros_like(image_gray)
# # 	# for kernel in filters:
# # 	filtered_image = cv2.filter2D(image_gray, cv2.CV_8UC3, kernel)
# # 	# np.maximum(image_total, filtered_image, image_total)
# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_4_" + str(numero_fruto) + version + name_file, filtered_image)

# # 	_, image_binary = cv2.threshold(filtered_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_4_bin_" + str(numero_fruto) + version + name_file, image_binary)

# # 	image_morph = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel)
# # 	image_morph = cv2.morphologyEx(image_morph, cv2.MORPH_CLOSE, kernel)
# # 	# cv2.imwrite(ruta_guardar+"danio_pudricion_4_" + str(numero_fruto) + version + name_file, image_morph)

# # 	glcm(image, image_morph, ruta_guardar, numero_fruto, name_file, "danio_pudricion_4")


# # def danio_pudricion_3_1(image,ruta_guardar,numero_fruto,name_file):
# # 	import pyfeats

# # 	features_mean, features_range, labels_mean, labels_range = glcm_features(image, ignore_zeros=True)

# # 	print(features_mean, features_range, labels_mean, labels_range)


# # def danio_pudricion_3(image,ruta_guardar,numero_fruto,name_file):
# 	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 	##
# 	from skimage.feature import greycomatrix, greycoprops
# 	from skimage import io, color, img_as_ubyte
# 	import matplotlib
# 	import matplotlib.pyplot as plt

# 	## 
# 	distances = [1, 2, 3]
# 	angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
# 	glcm = greycomatrix(image_gray, distances, angles, 256, symmetric=True, normed=True)
# 	# print(glcm)

# 	# Find the GLCM properties
# 	contrast = round(greycoprops(glcm, 'contrast')[0, 0],2)
# 	dissimilarity = round(greycoprops(glcm, 'dissimilarity')[0, 0],2)
# 	homogeneity = round(greycoprops(glcm, 'homogeneity')[0, 0],2)
# 	energy = round(greycoprops(glcm, 'energy')[0, 0],2)
# 	correlation = round(greycoprops(glcm, 'correlation')[0, 0],2)
# 	ASM = round(greycoprops(glcm, 'ASM')[0, 0],2)

# 	feature = [contrast, dissimilarity, homogeneity, energy, correlation, ASM]
# 	# print(feature)

# 	# xs = []
# 	# ys = []
# 	# xs.append(contrast)
# 	# ys.append(correlation)
	
# 	# fig = plt.figure()  # create a figure object
# 	# ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
# 	# marco_grafico = 20
# 	# ax.plot(xs[:marco_grafico], ys[:marco_grafico], 'go', label='Fruit')
# 	# ax.plot(xs[marco_grafico:], ys[marco_grafico:], 'bo', label='Defect')
# 	# ax.set_xlabel('GLCM Dissimilarity')
# 	# ax.set_ylabel('GLCM Correlation')
# 	# ax.legend()
# 	# fig.savefig(ruta_guardar + "GLMC_" + str(numero_fruto) + version + name_file, bbox_inches='tight')

# 	# return feature

# ## 
# def danio_pudricion_2_4(image,ruta_guardar,numero_fruto,name_file, pixeles_sum):	
# 	r, g, b = cv2.split(image)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_4_rgb_canal_R_" + str(numero_fruto) + version + name_file, r)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_rgb_canal_G_" + str(numero_fruto) + version + name_file, g)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_rgb_canal_B_" + str(numero_fruto) + version + name_file, b)

# 	image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# 	h, s, v = cv2.split(image_hsv)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hsv_canal_H_" + str(numero_fruto) + version + name_file, h)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hsv_canal_S_" + str(numero_fruto) + version + name_file, s)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_4_hsv_canal_V_" + str(numero_fruto) + version + name_file, v)

# 	image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
# 	l, a, b = cv2.split(image_lab)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_4_lab_canal_L_" + str(numero_fruto) + version + name_file, l)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_lab_canal_A_" + str(numero_fruto) + version + name_file, a)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_lab_canal_B_" + str(numero_fruto) + version + name_file, b)

# 	image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
# 	h, l2, s = cv2.split(image_hls)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hls_canal_H_" + str(numero_fruto) + version + name_file, h)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_4_hls_canal_L_" + str(numero_fruto) + version + name_file, l2)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hls_canal_S_" + str(numero_fruto) + version + name_file, s)

# 	image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
# 	ycrcb_y, ycrcb_cr, ycrcb_cb = cv2.split(image_hls)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_4_ycrcb_canal_ycrcb_y_" + str(numero_fruto) + version + name_file, ycrcb_y)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_ycrcb_canal_ycrcb_cr_" + str(numero_fruto) + version + name_file, ycrcb_cr)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_ycrcb_canal_ycrcb_cb_" + str(numero_fruto) + version + name_file, ycrcb_cb)

# 	## rgb, canal r buenos resultados
# 	## hsv, canal v resultados regulares
# 	## Mejorar imagen binaria con multiples imagenes de canales, eliminar contorno no daño, comparar resultados.
# 	_, image_binary_otsu_rgb_r = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 	_, image_binary_otsu_hsv_v = cv2.threshold(v, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 	_, image_binary_otsu_lab_l = cv2.threshold(l, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 	_, image_binary_otsu_hls_l = cv2.threshold(l2, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 	_, image_binary_otsu_ycrcb_y = cv2.threshold(ycrcb_y, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 	image_binary_otsu = cv2.bitwise_and(image_binary_otsu_rgb_r,image_binary_otsu_hsv_v)
# 	image_binary_otsu = cv2.bitwise_and(image_binary_otsu,image_binary_otsu_lab_l)
# 	image_binary_otsu = cv2.bitwise_and(image_binary_otsu,image_binary_otsu_hls_l)
# 	image_binary_otsu = cv2.bitwise_and(image_binary_otsu,image_binary_otsu_ycrcb_y)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_4_image_binary_otsu_" + str(numero_fruto) + version + name_file, image_binary_otsu)

# 	## 
# 	## image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_3_image_gray_" + str(numero_fruto) + version + name_file, image_gray)
	
# 	## _, mask_image_gray = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_3_mask_image_gray_" + str(numero_fruto) + version + name_file, mask_image_gray)

# 	## Contornos internos
# 	## findcontours
# 	_, contours, hierarchy = cv2.findContours(image=image_binary_otsu, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

# 	W, H = image.shape[:2]
# 	## create an empty mask
# 	mask_contour_internal = np.zeros((W, H),dtype=np.uint8)

# 	## Buscar como considerar daño oscuro (negro o cercano a este)
# 	for i, cnt in enumerate(contours):		
# 		## cv2.RETR_TREE
# 		## hierarchy = [[[A, B, C, D],[...],...]]: 
# 		## Jerarquia 0 (tupla 0 del arreglo hierarchy) donde: 
# 		## A siguiente contorno, B contornos previo, C contornos hijo (primero) y D contorno padre de C
# 		if hierarchy[0][i][2] >= 0 :
# 			# if the size of the contour is greater than a threshold
# 			# if cv2.contourArea(cnt) < 10:
# 			cv2.drawContours(mask_contour_internal, [cnt], 0, (255), -1)

# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_4_mask_contour_internal_" + str(numero_fruto) + version + name_file, mask_contour_internal)

# 	## Contornos externos
# 	## BUSCAR COMO OBTENER LOS CONTORNOS EXTERNOS DE UNA IMAGEN CEREZA.
# 	## findcontours
# 	_, contours, hierarchy = cv2.findContours(image=image_binary_otsu, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

# 	## create an empty mask
# 	mask_contour_external = np.zeros((W, H),dtype=np.uint8)

# 	for i, cnt in enumerate(contours):		
# 		if hierarchy[0][i][2] < 0:
# 			# if cv2.contourArea(cnt) > 5:
# 			cv2.drawContours(mask_contour_external, [cnt], 0, (255), -1)

# 	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_4_mask_contour_external_" + str(numero_fruto) + version + name_file, mask_contour_external)

# 	## CHEQUEAR COMO OPERAR LAS DISTITNAS MASCARA PARA OBTENER LA MAYOR CANTIDAD DE DANO DEL FRUTO Y MENOR CONTORNO EXTERNO.
# 	## Mejorar el calculo de mask
# 	mask_internal = cv2.bitwise_xor(image_binary_otsu,mask_contour_external)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_4_mask_internal_" + str(numero_fruto) + version + name_file, mask_internal)

# 	W, H = image.shape[:2]
# 	mask = np.zeros((W, H),dtype=np.uint8)
# 	# print(W,H,np.int8(W/2),np.int8(H/2),np.int8(W*(0.3)),np.int8(H*(0.3)),min(np.int8(W*(0.3)),np.int8(H*(0.3))))
# 	cv2.circle(mask,(np.int8(W/2),np.int8(H/2)),min(np.int8(W*(0.3)),np.int8(H*(0.3))),255,-1)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_4_mask_" + str(numero_fruto) + version + name_file, mask)

# 	mask_internal_reduce = cv2.bitwise_and(mask_internal, mask_internal, mask = mask)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_4_mask_internal_reduce_" + str(numero_fruto) + version + name_file, mask_internal_reduce)
# 	glcm(image, image_binary_otsu, ruta_guardar, numero_fruto, name_file, "danio_pudricion_2_3_mask_")

# 	# _, contours, _ = cv2.findContours(mask_internal_reduce, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 	# pixeles_r_cl_sum_cum = 0
# 	# pixeles_r_sum_cum = 0
# 	# pixeles_r_ca_sum_cum = 0
# 	# pixeles_ca_n_sum_cum = 0
# 	# pixeles_n_sum_cum = 0
# 	# for index, cnt in enumerate(contours):
# 	# 	x,y,w,h = cv2.boundingRect(cnt)
# 	# 	if (w*h > 9):
# 	# 		image_roi = image[y:y+h, x:x+w]
# 	# 		image_hsv = cv2.cvtColor(image_roi, cv2.COLOR_RGB2HSV)

# 	# 		## Toda la fruta
# 	# 		## mask_hsv = recognizer(image_hsv,[0,0,0],[255,255,255])
# 	# 		## roi_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask = None)

# 	# 		## pixeles = np.where(roi[:,:,0] > pixel_uno) and np.where(roi[:,:,1] > pixel_uno) and np.where(roi[:,:,2] > pixel_uno)
# 	# 		## pixeles_sum = (np.array(roi[pixeles])).sum()

# 	# 		## Canales: H, S y V, rangos color rojo claro
# 	# 		mask_hsv_r_cl = recognizer(image_hsv,[0,0,0],[20,255,255])
# 	# 		roi_r_cl = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_r_cl)
# 	# 		# cv2.imwrite(ruta_guardar+"roi_r_cl_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_r_cl, cv2.COLOR_HSV2BGR))
			
# 	# 		pixeles_r_cl = np.where(roi_r_cl[:,:,0] > pixel_uno) and np.where(roi_r_cl[:,:,1] > pixel_uno) and np.where(roi_r_cl[:,:,2] > pixel_uno)
# 	# 		pixeles_r_cl_sum = (np.array(roi_r_cl[pixeles_r_cl])).sum()

# 	# 		### Canales: H, S y V, rangos color rojo
# 	# 		mask_hsv_r = recognizer(image_hsv,[160,125,18],[180,205,100])
# 	# 		roi_r = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_r)
# 	# 		# cv2.imwrite(ruta_guardar+"roi_r_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_r, cv2.COLOR_HSV2RGB))

# 	# 		pixeles_r = np.where(roi_r[:,:,0] > pixel_uno) and np.where(roi_r[:,:,1] > pixel_uno) and np.where(roi_r[:,:,2] > pixel_uno)
# 	# 		pixeles_r_sum = (np.array(roi_r[pixeles_r])).sum()

# 	# 		## Canales: H, S y V, rangos color rojo caoba
# 	# 		mask_hsv_r_ca = recognizer(image_hsv,[160,0,0],[180,100,160])
# 	# 		roi_r_ca = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_r_ca)
# 	# 		# cv2.imwrite(ruta_guardar+"roi_r_ca_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_r_ca, cv2.COLOR_HSV2BGR))

# 	# 		pixeles_r_ca = np.where(roi_r_ca[:,:,0] > pixel_uno) and np.where(roi_r_ca[:,:,1] > pixel_uno) and np.where(roi_r_ca[:,:,2] > pixel_uno)
# 	# 		pixeles_r_ca_sum = (np.array(roi_r_ca[pixeles_r_ca])).sum()

# 	# 		# ## Canales: H, S y V, rangos color caoba
# 	# 		mask_hsv_ca_n = recognizer(image_hsv,[0,80,10],[125,125,100])
# 	# 		roi_r_ca_n = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_ca_n)
# 	# 		# cv2.imwrite(ruta_guardar+"roi_r_ca_n_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_r_ca_n, cv2.COLOR_HSV2RGB))

# 	# 		pixeles_ca_n = np.where(roi_r_ca_n[:,:,0] > pixel_uno) and np.where(roi_r_ca_n[:,:,1] > pixel_uno) and np.where(roi_r_ca_n[:,:,2] > pixel_uno)
# 	# 		pixeles_ca_n_sum = (np.array(roi_r_ca_n[pixeles_ca_n])).sum()

# 	# 		## Canales: H, S y V, rangos color negro
# 	# 		mask_hsv_n = recognizer(image_hsv,[0, 0, 0],[125, 125, 30])
# 	# 		roi_n = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_n)
# 	# 		# cv2.imwrite(ruta_guardar+"roi_n_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_n, cv2.COLOR_HSV2BGR))

# 	# 		pixeles_n = np.where(roi_n[:,:,0] > pixel_uno) and np.where(roi_n[:,:,1] > pixel_uno) and np.where(roi_n[:,:,2] > pixel_uno)
# 	# 		pixeles_n_sum = (np.array(roi_n[pixeles_n])).sum()

# 	# 		roi_all = cv2.add(roi_r_cl,roi_r)
# 	# 		roi_all = cv2.add(roi_all,roi_r_ca)
# 	# 		roi_all = cv2.add(roi_all,roi_r_ca_n)
# 	# 		roi_all = cv2.add(roi_all,roi_n)
# 	# 		# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_" + str(index) + "_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_all, cv2.COLOR_HSV2BGR))
			
# 	# 		pixeles_r_cl_sum_cum += pixeles_r_cl_sum
# 	# 		pixeles_r_sum_cum += pixeles_r_sum
# 	# 		pixeles_r_ca_sum_cum += pixeles_r_ca_sum
# 	# 		pixeles_ca_n_sum_cum += pixeles_ca_n_sum
# 	# 		pixeles_n_sum_cum += pixeles_n_sum

# 	# ##
# 	# if pixeles_r_cl_sum_cum != 0.0 or pixeles_r_sum_cum != 0.0 or pixeles_r_ca_sum_cum != 0.0 or pixeles_ca_n_sum_cum != 0.0 or pixeles_n_sum_cum != 0.0:
# 	# 	porcentaje_danio = round(100.0 - float(pixeles_sum - pixeles_r_cl_sum_cum - pixeles_r_sum_cum - pixeles_r_ca_sum_cum - pixeles_ca_n_sum_cum - pixeles_n_sum_cum) / float(pixeles_sum) * 100,1)
# 	# else:
# 	# 	porcentaje_danio = 0.0
		
# 	porcentaje_danio = 0
# 	return porcentaje_danio


# ## 
# def danio_pudricion_2_3(image,ruta_guardar,numero_fruto,name_file, pixeles_sum):	
# 	r, g, b = cv2.split(image)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_3_rgb_canal_R_" + str(numero_fruto) + version + name_file, r)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_rgb_canal_G_" + str(numero_fruto) + version + name_file, g)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_rgb_canal_B_" + str(numero_fruto) + version + name_file, b)

# 	image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# 	h, s, v = cv2.split(image_hsv)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hsv_canal_H_" + str(numero_fruto) + version + name_file, h)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hsv_canal_S_" + str(numero_fruto) + version + name_file, s)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_3_hsv_canal_V_" + str(numero_fruto) + version + name_file, v)

# 	image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
# 	l, a, b = cv2.split(image_lab)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_3_lab_canal_L_" + str(numero_fruto) + version + name_file, l)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_lab_canal_A_" + str(numero_fruto) + version + name_file, a)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_lab_canal_B_" + str(numero_fruto) + version + name_file, b)

# 	image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
# 	h, l2, s = cv2.split(image_hls)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hls_canal_H_" + str(numero_fruto) + version + name_file, h)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_3_hls_canal_L_" + str(numero_fruto) + version + name_file, l2)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hls_canal_S_" + str(numero_fruto) + version + name_file, s)

# 	image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
# 	ycrcb_y, ycrcb_cr, ycrcb_cb = cv2.split(image_hls)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_3_ycrcb_canal_ycrcb_y_" + str(numero_fruto) + version + name_file, ycrcb_y)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_ycrcb_canal_ycrcb_cr_" + str(numero_fruto) + version + name_file, ycrcb_cr)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_ycrcb_canal_ycrcb_cb_" + str(numero_fruto) + version + name_file, ycrcb_cb)

# 	## rgb, canal r buenos resultados
# 	## hsv, canal v resultados regulares
# 	## Mejorar imagen binaria con multiples imagenes de canales, eliminar contorno no daño, comparar resultados.
# 	_, image_binary_otsu_rgb_r = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 	_, image_binary_otsu_hsv_v = cv2.threshold(v, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 	_, image_binary_otsu_lab_l = cv2.threshold(l, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 	_, image_binary_otsu_hls_l = cv2.threshold(l2, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 	_, image_binary_otsu_ycrcb_y = cv2.threshold(ycrcb_y, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 	image_binary_otsu = cv2.bitwise_and(image_binary_otsu_rgb_r,image_binary_otsu_hsv_v)
# 	image_binary_otsu = cv2.bitwise_and(image_binary_otsu,image_binary_otsu_lab_l)
# 	image_binary_otsu = cv2.bitwise_and(image_binary_otsu,image_binary_otsu_hls_l)
# 	image_binary_otsu = cv2.bitwise_and(image_binary_otsu,image_binary_otsu_ycrcb_y)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_3_image_binary_otsu_" + str(numero_fruto) + version + name_file, image_binary_otsu)

# 	## 
# 	## image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_3_image_gray_" + str(numero_fruto) + version + name_file, image_gray)
	
# 	## _, mask_image_gray = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_3_mask_image_gray_" + str(numero_fruto) + version + name_file, mask_image_gray)

# 	W, H = image.shape[:2]
# 	mask = np.zeros((W, H),dtype=np.uint8)
# 	# print(W,H,np.int8(W/2),np.int8(H/2),np.int8(W*(0.3)),np.int8(H*(0.3)),min(np.int8(W*(0.3)),np.int8(H*(0.3))))
# 	cv2.circle(mask,(np.int8(W/2),np.int8(H/2)),min(np.int8(W*(0.3)),np.int8(H*(0.3))),255,-1)
# 	## cv2.imwrite(ruta_guardar+"danio_pudricion_2_3_mask_" + str(numero_fruto) + version + name_file, mask)

# 	image_binary_otsu = cv2.bitwise_and(image_binary_otsu, image_binary_otsu, mask = mask)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_3_image_binary_otsu_circle_" + str(numero_fruto) + version + name_file, image_binary_otsu)
# 	glcm(image, image_binary_otsu, ruta_guardar, numero_fruto, name_file, "danio_pudricion_2_3_mask_")

# 	# _, contours, _ = cv2.findContours(image_binary_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 	# pixeles_r_cl_sum_cum = 0
# 	# pixeles_r_sum_cum = 0
# 	# pixeles_r_ca_sum_cum = 0
# 	# pixeles_ca_n_sum_cum = 0
# 	# pixeles_n_sum_cum = 0
# 	# for index, cnt in enumerate(contours):
# 	# 	x,y,w,h = cv2.boundingRect(cnt)
# 	# 	if (w*h > 9):
# 	# 		image_roi = image[y:y+h, x:x+w]
# 	# 		image_hsv = cv2.cvtColor(image_roi, cv2.COLOR_RGB2HSV)

# 	# 		## Toda la fruta
# 	# 		## mask_hsv = recognizer(image_hsv,[0,0,0],[255,255,255])
# 	# 		## roi_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask = None)

# 	# 		## pixeles = np.where(roi_hsv[:,:,0] > pixel_uno) and np.where(roi_hsv[:,:,1] > pixel_uno) and np.where(roi_hsv[:,:,2] > pixel_uno)
# 	# 		## pixeles_sum = (np.array(roi_hsv[pixeles])).sum()

# 	# 		## Canales: H, S y V, rangos color rojo claro
# 	# 		mask_hsv_r_cl = recognizer(image_hsv,[0,0,0],[20,255,255])
# 	# 		roi_r_cl = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_r_cl)
# 	# 		## cv2.imwrite(ruta_guardar+"roi_r_cl_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_r_cl, cv2.COLOR_HSV2BGR))
			
# 	# 		pixeles_r_cl = np.where(roi_r_cl[:,:,0] > pixel_uno) and np.where(roi_r_cl[:,:,1] > pixel_uno) and np.where(roi_r_cl[:,:,2] > pixel_uno)
# 	# 		pixeles_r_cl_sum = (np.array(roi_r_cl[pixeles_r_cl])).sum()

# 	# 		### Canales: H, S y V, rangos color rojo
# 	# 		mask_hsv_r = recognizer(image_hsv,[160,125,18],[180,205,100])
# 	# 		roi_r = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_r)
# 	# 		## cv2.imwrite(ruta_guardar+"roi_r_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_r, cv2.COLOR_HSV2RGB))

# 	# 		pixeles_r = np.where(roi_r[:,:,0] > pixel_uno) and np.where(roi_r[:,:,1] > pixel_uno) and np.where(roi_r[:,:,2] > pixel_uno)
# 	# 		pixeles_r_sum = (np.array(roi_r[pixeles_r])).sum()

# 	# 		## Canales: H, S y V, rangos color rojo caoba
# 	# 		mask_hsv_r_ca = recognizer(image_hsv,[160,0,0],[180,100,160])
# 	# 		roi_r_ca = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_r_ca)
# 	# 		## cv2.imwrite(ruta_guardar+"roi_r_ca_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_r_ca, cv2.COLOR_HSV2BGR))

# 	# 		pixeles_r_ca = np.where(roi_r_ca[:,:,0] > pixel_uno) and np.where(roi_r_ca[:,:,1] > pixel_uno) and np.where(roi_r_ca[:,:,2] > pixel_uno)
# 	# 		pixeles_r_ca_sum = (np.array(roi_r_ca[pixeles_r_ca])).sum()

# 	# 		## Canales: H, S y V, rangos color caoba
# 	# 		mask_hsv_ca_n = recognizer(image_hsv,[0,80,10],[125,125,100])
# 	# 		roi_r_ca_n = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_ca_n)
# 	# 		## cv2.imwrite(ruta_guardar+"roi_r_ca_n_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_r_ca_n, cv2.COLOR_HSV2RGB))

# 	# 		pixeles_ca_n = np.where(roi_r_ca_n[:,:,0] > pixel_uno) and np.where(roi_r_ca_n[:,:,1] > pixel_uno) and np.where(roi_r_ca_n[:,:,2] > pixel_uno)
# 	# 		pixeles_ca_n_sum = (np.array(roi_r_ca_n[pixeles_ca_n])).sum()

# 	# 		## Canales: H, S y V, rangos color negro
# 	# 		mask_hsv_n = recognizer(image_hsv,[0, 0, 0],[125, 125, 30])
# 	# 		roi_n = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_n)
# 	# 		## cv2.imwrite(ruta_guardar+"roi_n_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_n, cv2.COLOR_HSV2BGR))

# 	# 		pixeles_n = np.where(roi_n[:,:,0] > pixel_uno) and np.where(roi_n[:,:,1] > pixel_uno) and np.where(roi_n[:,:,2] > pixel_uno)
# 	# 		pixeles_n_sum = (np.array(roi_n[pixeles_n])).sum()

# 	# 		roi_all = cv2.add(roi_r_cl,roi_r)
# 	# 		roi_all = cv2.add(roi_all,roi_r_ca)
# 	# 		roi_all = cv2.add(roi_all,roi_r_ca_n)
# 	# 		roi_all = cv2.add(roi_all,roi_n)
# 	# 		## cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_" + str(index) + "_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_all, cv2.COLOR_HSV2BGR))
			
# 	# 		pixeles_r_cl_sum_cum += pixeles_r_cl_sum
# 	# 		pixeles_r_sum_cum += pixeles_r_sum
# 	# 		pixeles_r_ca_sum_cum += pixeles_r_ca_sum
# 	# 		pixeles_ca_n_sum_cum += pixeles_ca_n_sum
# 	# 		pixeles_n_sum_cum += pixeles_n_sum

# 	# ##
# 	# if pixeles_r_cl_sum_cum != 0.0 or pixeles_r_sum_cum != 0.0 or pixeles_r_ca_sum_cum != 0.0 or pixeles_ca_n_sum_cum != 0.0 or pixeles_n_sum_cum != 0.0:
# 	# 	porcentaje_danio = round(100.0 - float(pixeles_sum - pixeles_r_cl_sum_cum - pixeles_r_sum_cum - pixeles_r_ca_sum_cum - pixeles_ca_n_sum_cum - pixeles_n_sum_cum) / float(pixeles_sum) * 100,1)
# 	# else:
# 	# 	porcentaje_danio = 0.0
		
# 	porcentaje_danio = 0
# 	return porcentaje_danio


def danio_pudricion_2_2(image,ruta_guardar,numero_fruto,name_file, pixeles_sum):	
	## Calculo total de pixeles del arandano
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_" + str(numero_fruto) + version + name_file, image_binary)

	image_binary_not = cv2.bitwise_not(image_binary)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_not_" + str(numero_fruto) + version + name_file, image_binary_not)

	## Total de pixeles del arandano
	pixeles_image_binary_not = np.where(image_binary_not[:,:] > pixel_uno)
	pixeles_image_binary_not_sum = (np.array(image_binary_not[pixeles_image_binary_not])).sum()

	##
	r, g, b = cv2.split(image)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_rgb_canal_R_" + str(numero_fruto) + version + name_file, r)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_rgb_canal_G_" + str(numero_fruto) + version + name_file, g)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_rgb_canal_B_" + str(numero_fruto) + version + name_file, b)

	##
	image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	h, s, v = cv2.split(image_hsv)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hsv_canal_H_" + str(numero_fruto) + version + name_file, h)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hsv_canal_S_" + str(numero_fruto) + version + name_file, s)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hsv_canal_V_" + str(numero_fruto) + version + name_file, v)

	## Detecta ausencia de cera por manipulacion
	_, image_binary_otsu_hsv_s = cv2.threshold(s, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_otsu_hsv_s_" + str(numero_fruto) + version + name_file, image_binary_otsu_hsv_s)

	##
	image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
	l, a, b = cv2.split(image_lab)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_lab_canal_L_" + str(numero_fruto) + version + name_file, l)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_lab_canal_A_" + str(numero_fruto) + version + name_file, a)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_lab_canal_B_" + str(numero_fruto) + version + name_file, b)

	## 
	image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	h, l, s2 = cv2.split(image_hls)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hls_canal_H_" + str(numero_fruto) + version + name_file, h)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hls_canal_L_" + str(numero_fruto) + version + name_file, l)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_hls_canal_S_" + str(numero_fruto) + version + name_file, s2)

	## Detecta fruta sin cera natural (resultados regulares)
	_, image_binary_otsu_hls_s2 = cv2.threshold(s2, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_otsu_hls_s2_" + str(numero_fruto) + version + name_file, image_binary_otsu_hls_s2)

	## Detecta fruta sin cera natural invertida (resultados regulares)
	image_binary_otsu_hls_s2_not = cv2.bitwise_not(image_binary_otsu_hls_s2)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_otsu_hls_s2_not_" + str(numero_fruto) + version + name_file, image_binary_otsu_hls_s2_not)

	## Detecta fruta sin cera natural y deshidratacion usando mascara (resultados regulares)
	image_binary_otsu_hls_s2_mask = cv2.bitwise_and(image_binary_otsu_hls_s2,image_binary_otsu_hls_s2,mask=image_binary_not)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_otsu_hls_s2_mask_" + str(numero_fruto) + version + name_file, image_binary_otsu_hls_s2_mask)

	## Calculo de ausencia de cera
	# sin_cera = cv2.bitwise_and(image_binary_otsu_hls_s2_mask,image_binary_otsu_hls_s2_mask,mask=image_binary_otsu_hsv_s)
	sin_cera = cv2.bitwise_and(image_binary_otsu_hsv_s,image_binary_otsu_hls_s2_mask)
	# sin_cera_ero = cv2.erode(sin_cera, kernel_elipse, iterations=1)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_sin_cera_" + str(numero_fruto) + version + name_file, sin_cera)

	pixeles_sin_cera = np.where(sin_cera[:,:] > pixel_uno)
	pixeles_sin_cera_sum = (np.array(sin_cera[pixeles_sin_cera])).sum()

	## Sin cera invertido
	sin_cera_not = cv2.bitwise_not(sin_cera)
	sin_cera_not = cv2.bitwise_and(sin_cera_not,image_binary_not)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_sin_cera_not_" + str(numero_fruto) + version + name_file, sin_cera_not)
	pixeles_sin_cera_not = np.where(sin_cera_not[:,:] > pixel_uno)
	pixeles_sin_cera_not_sum = (np.array(sin_cera_not[pixeles_sin_cera_not])).sum()

	##
	image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
	y, cr, cb = cv2.split(image_ycrcb)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_ycrcb_canal_y_" + str(numero_fruto) + version + name_file, y)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_ycrcb_canal_cr_" + str(numero_fruto) + version + name_file, cr)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_ycrcb_canal_cb_" + str(numero_fruto) + version + name_file, cb)

	_, image_binary_otsu_ycrcb_cb = cv2.threshold(cb, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_otsu_ycrcb_cb_" + str(numero_fruto) + version + name_file, image_binary_otsu_ycrcb_cb)

	## Detecta fruta con daño + contornos deshidratados
	image_binary_otsu_ycrcb_cb_not = cv2.bitwise_not(image_binary_otsu_ycrcb_cb)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_otsu_ycrcb_cb_not_" + str(numero_fruto) + version + name_file, image_binary_otsu_ycrcb_cb_not)

	## Detecta fruta con daño + contornos deshidratados usando mascara
	image_binary_otsu_ycrcb_cb_not_mask = cv2.bitwise_and(image_binary_otsu_ycrcb_cb_not, image_binary_otsu_ycrcb_cb_not, mask = image_binary_not)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_otsu_ycrcb_cb_not_mask_" + str(numero_fruto) + version + name_file, image_binary_otsu_ycrcb_cb_not_mask)

	# image_binary_otsu_ycrcb_cb_not_mask_opening = cv2.morphologyEx(image_binary_otsu_ycrcb_cb_not_mask, cv2.MORPH_OPEN, kernel_elipse)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_otsu_ycrcb_cb_not_mask_opening_" + str(numero_fruto) + version + name_file, image_binary_otsu_ycrcb_cb_not_mask_opening)

	## Calculamos pixeles dañados + deshidratados
	pixeles_daniado_deshidratado = np.where(image_binary_otsu_ycrcb_cb_not_mask[:,:] > pixel_uno)
	pixeles_daniado_deshidratado_sum = (np.array(image_binary_otsu_ycrcb_cb_not_mask[pixeles_daniado_deshidratado])).sum()
	
	# image_binary_otsu_hsv_s_hls_s2 = cv2.bitwise_and(image_binary_otsu_hsv_s,image_binary_otsu_hls_s2)
	# image_binary_otsu_hsv_s_hls_s2_ycrcb_cb = cv2.bitwise_and(image_binary_otsu_hsv_s_hls_s2,image_binary_otsu_ycrcb_cb)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_otsu_hsv_s_hls_s2_ycrcb_cb_" + str(numero_fruto) + version + name_file, image_binary_otsu_hsv_s_hls_s2_ycrcb_cb)

	# ## Convolucion
	# ## Low pass
	# kernel2 = np.array([[1, 1, 1, 1, 1], 
	# 				[1, 1, 1, 1, 1], 
	# 				[1, 1, 1, 1, 1], 
	# 				[1, 1, 1, 1, 1], 
	# 				[1, 1, 1, 1, 1]])
	# kernel3 = kernel2/sum(kernel2)

	# ## filter the source image
	# image_conv_lp = cv2.filter2D(image,-1,kernel3)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_conv_lp_" + str(numero_fruto) + version + name_file, image_conv_lp)

	# # USAR image_conv_lp para calculo de danio, donde el blanco o cercano al este es danio del fruto

	# ## High pass
	# #edge detection filter
	# # kernel2 = np.array([[0.0, -1.0, 0.0], 
	# # 				[-1.0, 4.0, -1.0],
	# # 				[0.0, -1.0, 0.0]])
	# kernel2 = np.array([[0, 0, 1, 0, 0], 
	# 					[0, 1, 2, 1, 0], 
	# 					[1, 2, 3, 2, 1], 
	# 					[0, 1, 2, 1, 0], 
	# 					[0, 0, 1, 0, 0]])
	# kernel3 = kernel2/(np.sum(kernel2) if np.sum(kernel2)!=0 else 1)

	# ## filter the source image
	# image_conv_hp = cv2.filter2D(image,-1,kernel3)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_conv_high_pass_" + str(numero_fruto) + version + name_file, image_conv_hp)

	# ## Gradiente
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_gray_" + str(numero_fruto) + version + name_file, gray)

	# # compute gradients along the x and y axis, respectively
	# gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
	# gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
	
	# # compute the gradient magnitude and orientation
	# magnitude = np.sqrt((gX ** 2) + (gY ** 2))
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_grad_magnitude_" + str(numero_fruto) + version + name_file, magnitude)

	# orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_grad_orientation_" + str(numero_fruto) + version + name_file, orientation)

	# ## Bilateral Filter
	# image_bilateral_filter = cv2.bilateralFilter(src=image, d=9, sigmaColor=75, sigmaSpace=75)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_bilateral_filter_" + str(numero_fruto) + version + name_file, image_bilateral_filter)

	# ##
	# image_conv_lp_gray = cv2.cvtColor(image_conv_lp, cv2.COLOR_BGR2GRAY)
	# ## _, image_conv_lp_binary = cv2.threshold(image_conv_lp_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# _, image_conv_lp_binary = cv2.threshold(image_conv_lp_gray, 253, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_conv_lp_binary_" + str(numero_fruto) + version + name_file, image_conv_lp_binary)

	# ## Detecta ausencia de cera por manipulacion full
	# image_conv_lp_binary_not = cv2.bitwise_not(image_conv_lp_binary)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_conv_lp_binary_not_" + str(numero_fruto) + version + name_file, image_conv_lp_binary_not)

	# image_bilateral_filter_gray = cv2.cvtColor(image_bilateral_filter, cv2.COLOR_BGR2GRAY)
	# ## _, image_bilateral_filter_binary = cv2.threshold(image_bilateral_filter_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# _, image_bilateral_filter_binary = cv2.threshold(image_bilateral_filter_gray, 253, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_bilateral_filter_binary_" + str(numero_fruto) + version + name_file, image_bilateral_filter_binary)

	# image_bilateral_filter_binary_not = cv2.bitwise_not(image_bilateral_filter_binary)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_bilateral_filter_binary_not_" + str(numero_fruto) + version + name_file, image_bilateral_filter_binary_not)

	# ## calcula ausencia de cera por manipulacion full
	# image_binary_bitwise = cv2.bitwise_and(image_conv_lp_binary_not,image_bilateral_filter_binary_not)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_bitwise_" + str(numero_fruto) + version + name_file, image_binary_bitwise)

	# image_binary = cv2.bitwise_and(image_bilateral_filter_binary_not,image_conv_lp_binary)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_binary_" + str(numero_fruto) + version + name_file, image_binary)

	## image_binary_otsu_ycrcb_cb_not + image_binary_otsu_hls_s2_not (solo fruta con danio)
	# image_result = cv2.bitwise_and(image_binary_otsu_ycrcb_cb_not,image_binary_otsu_hls_s2_not, mask = image_binary_not)
	# image_result = cv2.bitwise_and(image_binary_otsu_ycrcb_cb_not,image_binary_otsu_ycrcb_cb_not, mask = image_binary_not)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_image_result_" + str(numero_fruto) + version + name_file, image_result)

	## usar resulatdo image_conv_lp para fruta deshidratada
	## combinar o sumar ambas cantidades de pixeles (image_result y image_conv_lp) para calcular el % de daño

	##
	# print(pixeles_image_binary_otsu_ycrcb_cb_not_mask_sum,pixeles_image_binary_otsu_hls_s2_sum,pixeles_image_binary_not_sum)
	# porcentaje_danio = round((pixeles_image_binary_otsu_ycrcb_cb_not_mask_sum + pixeles_image_binary_otsu_hls_s2_sum) / pixeles_image_binary_not_sum * 100,2)
	# porcentaje_danio = round(pixeles_image_binary_otsu_ycrcb_cb_not_mask_sum / pixeles_image_binary_not_sum * 100,2)

	# _, image_tozero_inv = cv2.threshold(image_gray, 127, 255, cv2.THRESH_TOZERO_INV)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_2_" + str(numero_fruto) + version + name_file, image_tozero_inv)

	# print('daño')
	# print(pixeles_daniado_deshidratado_sum, pixeles_image_binary_not_sum)
	porcentaje_danio = round(100 * pixeles_daniado_deshidratado_sum/pixeles_image_binary_not_sum,2)
	# porcentaje_manipulacion = round(100 * pixeles_sin_cera_sum/pixeles_image_binary_not_sum,2)
	# print('manipulación')
	# print(pixeles_sin_cera_not_sum, pixeles_image_binary_not_sum)
	porcentaje_manipulacion = round(100 * pixeles_sin_cera_not_sum/pixeles_image_binary_not_sum,2)

	# ##
	# if pixeles_r_cl_sum_cum != 0.0 or pixeles_r_sum_cum != 0.0 or pixeles_r_ca_sum_cum != 0.0 or pixeles_ca_n_sum_cum != 0.0 or pixeles_n_sum_cum != 0.0:
	# 	porcentaje_danio = round(100.0 - float(pixeles_sum - pixeles_r_cl_sum_cum - pixeles_r_sum_cum - pixeles_r_ca_sum_cum - pixeles_ca_n_sum_cum - pixeles_n_sum_cum) / float(pixeles_sum) * 100,1)
	# else:
	# 	porcentaje_danio = 0.0

	# if porcentaje_danio < 20.0:
	# 	porcentaje_danio = 0.0
	
	# if porcentaje_manipulacion < 70.0:
	# 	porcentaje_manipulacion = 0.0
	
	# porcentaje_danio = 0.0
	return [porcentaje_danio,porcentaje_manipulacion]



## Solo usamos gray como input.
def danio_pudricion_2_1(image,ruta_guardar,numero_fruto,name_file, pixeles_sum):	
	## Calculo total de pixeles del arandano
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_image_binary_" + str(numero_fruto) + version + name_file, image_binary)

	image_binary_not = cv2.bitwise_not(image_binary)
	cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_image_binary_not_" + str(numero_fruto) + version + name_file, image_binary_not)

	image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	h, l, s2 = cv2.split(image_hls)
	# # cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_hls_canal_H_" + str(numero_fruto) + version + name_file, h)
	# # cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_hls_canal_L_" + str(numero_fruto) + version + name_file, l)
	cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_hls_canal_S_" + str(numero_fruto) + version + name_file, s2)

	# ## Detecta fruta sin cera natural (no da buenos resultados)
	_, image_binary_otsu_hls_s2 = cv2.threshold(s2, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_image_binary_otsu_hls_s2_" + str(numero_fruto) + version + name_file, image_binary_otsu_hls_s2)

	image_binary_otsu_hls_s2_not = cv2.bitwise_not(image_binary_otsu_hls_s2)
	cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_image_binary_otsu_hls_s2_not_" + str(numero_fruto) + version + name_file, image_binary_otsu_hls_s2_not)

	## Detecta defecto
	image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
	y, cr, cb = cv2.split(image_ycrcb)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_ycrcb_canal_y_" + str(numero_fruto) + version + name_file, y)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_ycrcb_canal_cr_" + str(numero_fruto) + version + name_file, cr)
	cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_ycrcb_canal_cb_" + str(numero_fruto) + version + name_file, cb)

	_, image_binary_otsu_ycrcb_cb = cv2.threshold(cb, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_image_binary_otsu_ycrcb_cb_" + str(numero_fruto) + version + name_file, image_binary_otsu_ycrcb_cb)

	## Detecta fruta con daño
	image_binary_otsu_ycrcb_cb_not = cv2.bitwise_not(image_binary_otsu_ycrcb_cb)
	cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_image_binary_otsu_ycrcb_cb_not_" + str(numero_fruto) + version + name_file, image_binary_otsu_ycrcb_cb_not)

	## Contamos pixeles del canal cb del espacio de color YCrCb
	image_binary_otsu_ycrcb_cb_not_mask = cv2.bitwise_and(image_binary_otsu_ycrcb_cb_not, image_binary_otsu_ycrcb_cb_not, mask = image_binary_not)
	cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_image_binary_otsu_ycrcb_cb_not_mask_" + str(numero_fruto) + version + name_file, image_binary_otsu_ycrcb_cb_not_mask)

	pixeles_image_binary_otsu_ycrcb_cb_not_mask = np.where(image_binary_otsu_ycrcb_cb_not_mask[:,:] > pixel_uno)
	pixeles_image_binary_otsu_ycrcb_cb_not_mask_sum = (np.array(image_binary_otsu_ycrcb_cb_not_mask[pixeles_image_binary_otsu_ycrcb_cb_not_mask])).sum()

	## image_binary_otsu_ycrcb_cb_not (solo fruta con danio)
	image_result = cv2.bitwise_and(image_binary_otsu_ycrcb_cb_not,image_binary_otsu_ycrcb_cb_not, mask = image_binary_not)
	cv2.imwrite(ruta_guardar+"danio_pudricion_2_1_image_result_" + str(numero_fruto) + version + name_file, image_result)

	## usar resulatdo image_conv_lp para fruta deshidratada
	## combinar o sumar ambas cantidades de pixeles (image_result y image_conv_lp) para calcular el % de daño

	## Total de pixeles del arandano
	pixeles_image_binary_not = np.where(image_binary_not[:,:] > pixel_uno)
	pixeles_image_binary_not_sum = (np.array(image_binary_not[pixeles_image_binary_not])).sum()

	##
	porcentaje_danio = round(pixeles_image_binary_otsu_ycrcb_cb_not_mask_sum / pixeles_image_binary_not_sum * 100,2)

	return porcentaje_danio


# def danio_pudricion_2(image,ruta_guardar,numero_fruto,name_file, pixeles_sum):
# 	image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# 	# ## Toda la fruta
# 	# mask = recognizer(image_hsv,[0,0,0],[255,255,255])
# 	# roi = cv2.bitwise_and(image_hsv, image_hsv, mask = None)
# 	# cv2.imwrite(ruta_guardar+"danio_pudricion_2_roi_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi, cv2.COLOR_HSV2BGR))

# 	# pixeles = np.where(roi[:,:,0] != pixel_negro) and np.where(roi[:,:,1] != pixel_negro) and np.where(roi[:,:,2] != pixel_negro)
# 	# pixeles_sum = (np.array(roi[pixeles])).sum()

# 	## Canales: H, S y V, rangos color rojo claro
# 	mask_hsv_r_cl = recognizer(image_hsv,[0,0,0],[20,255,255])
# 	roi_r_cl = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_r_cl)
# 	# cv2.imwrite(ruta_guardar+"roi_r_cl_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_r_cl, cv2.COLOR_HSV2BGR))
	
# 	pixeles_r_cl = np.where(roi_r_cl[:,:,0] != pixel_negro) and np.where(roi_r_cl[:,:,1] != pixel_negro) and np.where(roi_r_cl[:,:,2] != pixel_negro)
# 	pixeles_r_cl_sum = (np.array(roi_r_cl[pixeles_r_cl])).sum()

# 	### Canales: H, S y V, rangos color rojo
# 	mask_hsv_r = recognizer(image_hsv,[160,125,18],[180,205,100])
# 	roi_r = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_r)
# 	# cv2.imwrite(ruta_guardar+"roi_r_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_r, cv2.COLOR_HSV2RGB))

# 	pixeles_r = np.where(roi_r[:,:,0] != pixel_negro) and np.where(roi_r[:,:,1] != pixel_negro) and np.where(roi_r[:,:,2] != pixel_negro)
# 	pixeles_r_sum = (np.array(roi_r[pixeles_r])).sum()

# 	## Canales: H, S y V, rangos color rojo caoba
# 	mask_hsv_r_ca = recognizer(image_hsv,[160,0,0],[180,100,160])
# 	roi_r_ca = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_r_ca)
# 	# cv2.imwrite(ruta_guardar+"roi_r_ca_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_r_ca, cv2.COLOR_HSV2BGR))

# 	pixeles_r_ca = np.where(roi_r_ca[:,:,0] != pixel_negro) and np.where(roi_r_ca[:,:,1] != pixel_negro) and np.where(roi_r_ca[:,:,2] != pixel_negro)
# 	pixeles_r_ca_sum = (np.array(roi_r_ca[pixeles_r_ca])).sum()

# 	# ## Canales: H, S y V, rangos color caoba
# 	mask_hsv_ca_n = recognizer(image_hsv,[0,80,10],[125,125,100])
# 	roi_r_ca_n = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_ca_n)
# 	# cv2.imwrite(ruta_guardar+"roi_r_ca_n_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_r_ca_n, cv2.COLOR_HSV2RGB))

# 	pixeles_ca_n = np.where(roi_r_ca_n[:,:,0] != pixel_negro) and np.where(roi_r_ca_n[:,:,1] != pixel_negro) and np.where(roi_r_ca_n[:,:,2] != pixel_negro)
# 	pixeles_ca_n_sum = (np.array(roi_r_ca_n[pixeles_ca_n])).sum()

# 	## Canales: H, S y V, rangos color negro
# 	mask_hsv_n = recognizer(image_hsv,[0, 0, 0],[125, 125, 30])
# 	roi_n = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_hsv_n)
# 	# cv2.imwrite(ruta_guardar+"roi_n_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_n, cv2.COLOR_HSV2BGR))

# 	pixeles_n = np.where(roi_n[:,:,0] != pixel_negro) and np.where(roi_n[:,:,1] != pixel_negro) and np.where(roi_n[:,:,2] != pixel_negro)
# 	pixeles_n_sum = (np.array(roi_n[pixeles_n])).sum()

# 	roi_all = cv2.add(roi_r_cl,roi_r)
# 	roi_all = cv2.add(roi_all,roi_r_ca)
# 	roi_all = cv2.add(roi_all,roi_r_ca_n)
# 	roi_all = cv2.add(roi_all,roi_n)
# 	cv2.imwrite(ruta_guardar+"danio_pudricion_2_roi_all_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_all, cv2.COLOR_HSV2BGR))
	
# 	##
# 	if pixeles_r_cl_sum != 0.0 or pixeles_r_sum != 0.0 or pixeles_r_ca_sum != 0.0 or pixeles_ca_n_sum != 0.0 or pixeles_n_sum != 0.0:
# 		porcentaje_danio = round(100.0 - float(pixeles_sum - pixeles_r_cl_sum - pixeles_r_sum - pixeles_r_ca_sum - pixeles_ca_n_sum - pixeles_n_sum) / float(pixeles_sum) * 100,1)
# 	else:
# 		porcentaje_danio = 0.0
	
# 	return porcentaje_danio


def danio_pudricion_1(image,ruta_guardar,numero_fruto,name_file):
	image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

	## Toda la fruta
	mask = recognizer(image_hsv,[0,0,0],[255,255,255])
	roi = cv2.bitwise_and(image_hsv, image_hsv, mask = None)

	pixeles = np.where(roi[:,:,0] != pixel_negro) and np.where(roi[:,:,1] != pixel_negro) and np.where(roi[:,:,2] != pixel_negro)
	pixeles_sum = (np.array(roi[pixeles])).sum()

	## Fruta con danio
	mask_danio = recognizer(image_hsv,[72,60,72],[80,80,80]) # mascara de la zona danada color
	# cv2.imwrite(ruta_guardar+"mask_danio_" + str(numero_fruto) + version + name_file, mask_danio)
	roi_danio = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_danio)
	
	pixeles_danio = np.where(roi_danio[:,:,0] != pixel_negro) and np.where(roi_danio[:,:,1] != pixel_negro) and np.where(roi_danio[:,:,2] != pixel_negro)
	pixeles_danio_sum = (np.array(roi_danio[pixeles_danio])).sum()

	## Fruta con pudricion
	mask_pudricion = recognizer(image_hsv,[20,30,30],[40,50,50]) # mascara de la zona danada pudricion
	# cv2.imwrite(ruta_guardar+"mask_pudricion_" + str(numero_fruto) + version + name_file, mask_pudricion)
	roi_pudricion= cv2.bitwise_and(image_hsv, image_hsv, mask = mask_pudricion)
	
	pixeles_pudricion = np.where(roi_pudricion[:,:,0] != pixel_negro) and np.where(roi_pudricion[:,:,1] != pixel_negro) and np.where(roi_pudricion[:,:,2] != pixel_negro)
	pixeles_pudricion_sum = (np.array(roi_pudricion[pixeles_pudricion])).sum()

	## Fruta deshidratada
	mask_deshidratada = recognizer(image_hsv,[164,22,28],[165,58,49])
	# cv2.imwrite(ruta_guardar+"mask_deshidratada_" + str(numero_fruto) + version + name_file, mask_deshidratada)
	roi_deshidratada = cv2.bitwise_and(image_hsv, image_hsv, mask = mask_deshidratada)

	pixeles_deshidratada = np.where(roi_deshidratada[:,:,0] != pixel_negro) and np.where(roi_deshidratada[:,:,1] != pixel_negro) and np.where(roi_deshidratada[:,:,2] != pixel_negro)
	pixeles_deshidratada_sum = (np.array(roi_deshidratada[pixeles_deshidratada])).sum()

	roi_all = cv2.add(roi_danio,roi_pudricion)
	roi_all = cv2.add(roi_all,roi_deshidratada)
	# cv2.imwrite(ruta_guardar+"danio_pudricion_1_" + str(numero_fruto) + version + name_file, cv2.cvtColor(roi_all, cv2.COLOR_HSV2BGR))
	
	##
	if pixeles_danio_sum != 0.0 or pixeles_pudricion_sum != 0.0 or pixeles_deshidratada_sum != 0.0:
		porcentaje_danio = round(float(pixeles_danio_sum + pixeles_pudricion_sum + pixeles_deshidratada_sum) / float(pixeles_sum) * 100,1)
	else:
		porcentaje_danio = 0.0

	return porcentaje_danio


## --------------------------------------------------------------------------------------------------------------------

# # USAGE
# # python detect_color.py -i 1.jpg

# #~ --------------------------------------
# #~ Import package
# #~ --------------------------------------
# from declarar_librerias import *

# def recognizer(image,low,hi):
# 	#~ Array definition between 2 RGB colors / se define los parametros entre los cuales se considera la matriz
# 	lower = np.array(low, dtype = "uint8") #Se define el limite inferior en RGB
# 	upper = np.array(hi, dtype = "uint8")  #Se define el limite superior en RGB
# 	#~ Mask creation / Se crea la mascara
# 	mask = cv2.inRange(image, lower, upper)

# 	#~ Mask relaxation / La difinicion de la mascara se relaja y se difumina para cubrir correctamente el area al rededor de los pixeles identificados
# 	kernel = np.ones((5,5),np.float32)/8
# 	dst = cv2.filter2D(mask,-3,kernel)
# 	(thresh, im_bw) = cv2.threshold(dst, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 	#~ Image output / se genera un ouput de imagen que puede ser visualizado
# 	output = cv2.bitwise_and(image, image, mask = im_bw)
# 	return output, im_bw

# #~ --------------------------------------
# #~ ArgumentParser
# #~ --------------------------------------
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image")
# args = vars(ap.parse_args())

# image = cv2.imread(args["image"]) #se lee el nombre del archivo

# def danio_pudricion(image):
# 	#~ image = cv2.imread("1.jpg")
# 	output5, color = recognizer(image,[72,60,72],[80,80,80]) # mascara de la zona danada color
# 	output, im_bw = recognizer(image,[20,30,30],[40,50,50]) # mascara de la zona danada pudricion
# 	output2, im_bw2 = recognizer(image,[10,10,10],[150,90,170]) # mascara de la zona sana
# 	output3, im_bw3 = recognizer(image,[150,150,150],[255,255,255]) # mascara de la zona en blanco
# 	output4, total = recognizer(image,[0,0,0],[255,255,255])

# 	#~ --------------------------------------
# 	#~ Image output
# 	#~ --------------------------------------

# 	#plt.imshow(output5)
# 	#plt.show()

# 	#~ --------------------------------------
# 	#~ Statistics
# 	#~ --------------------------------------
# 	damaged=cv2.countNonZero(im_bw) #Se calcula la cantidad de pixeles no Negros en la mascara
# 	damaged_color = cv2.countNonZero(color)
# 	#sana=cv2.countNonZero(im_bw2)  #Se calcula la cantidad de pixeles no Negros en la mascara
# 	#white=cv2.countNonZero(im_bw3)   #Se calcula la cantidad de pixeles no Negros en la mascara
# 	total2 = cv2.countNonZero(total)
# 	pudricion = 100*damaged/float(total2)
# 	color = 100*damaged_color/float(total2)
# 	if (color >= 12 or pudricion >= 12):
# 		if (color > pudricion):
# 			return (color-12)
# 		else:
# 			return (pudricion - 12)
# 	else:
# 		color = 0.0
# 		return color
# 	#fruto = total2 - white
# 	#falta_pix = fruto - sana