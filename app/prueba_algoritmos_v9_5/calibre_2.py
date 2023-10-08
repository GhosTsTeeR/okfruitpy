from app.declarar_librerias import * 
from declarar_variables import *

def calibre_2(img, img_mask, numero_fruto, name_file, ruta_guardar): 
    ## 
    img_copy = img.copy()

    ## 
    _, contours, hierachy= cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = img_copy
    w_elipse = 0
    h_elipse = 0
    radius_elipse = 0
    angulo_elipse = 0
    for index, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        if (len(cnt) >= 5):
            
            (xc,xy),(a,b),angulo = cv2.fitEllipse(cnt)
            # print(xc,xy,a,b,angulo)
            # print(math.isnan(xc), math.isinf(xc), math.isnan(xy), math.isinf(xy), math.isnan(a), math.isinf(a), math.isnan(b), math.isinf(b), math.isnan(angulo), math.isinf(angulo) )
            
            if( not (math.isnan(xc) or math.isinf(xc) or math.isnan(xy) or math.isinf(xy) or math.isnan(a) or math.isinf(a) or math.isnan(b) or math.isinf(b) or math.isnan(angulo) or math.isinf(angulo) ) ):
                
                # cv2.ellipse(img,((xc,xy),(a,b),angulo),(0,255,0),1)
                # cv2.imwrite(ruta_guardar+"img_fitEllipse_"+ str(numero_fruto) + version + name_file, img)

                # (xc,xy),(a,b),angulo= cv2.fitEllipse(cnt)
                
                if (radius_elipse < b):
                    radius_elipse = b
                
                if (angulo_elipse < angulo):
                    angulo_elipse = angulo

                if (w_elipse < w):
                    w_elipse = w

                if (h_elipse < h):
                    h_elipse = h

    ##
    img = img_copy
    radius_hough = 0
    w_hough = 0
    h_hough = 0
    circles = cv2.HoughCircles(img_mask, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=100, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius_aux = i[2]

            if radius_hough < radius_aux:
                radius_hough = radius_aux

            if(w_hough < i[0]):
                w_hough = i[0]

            if(h_hough < i[1]):
                h_hough = i[1]

            # cv2.circle(img, center, radius_aux, (0,255,0), 1)
            # cv2.imwrite(ruta_guardar+"img_hough_circle_"+ str(numero_fruto) + version + name_file, img)
    

    ##
    img = img_copy
    cnts = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    radius_enclosing = 0
    w_enclosing = 0
    h_enclosing = 0
    for c in cnts:
        (x, y), r = cv2.minEnclosingCircle(c)

        if(radius_enclosing < r):
            radius_enclosing = r

        if(w_enclosing < x):
            w_enclosing = x
        if(h_enclosing < y):
            h_enclosing = y

        # cv2.circle(img, (int(x), int(y)), int(r), (0,255,0), 1)
        # cv2.imwrite(ruta_guardar+"img_minEnclosing_"+ str(numero_fruto) + version + name_file, img)
    
    # ## 
    # img = img_copy
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # ret,thresh = cv2.threshold(img,127,255,0)
    # img = img_copy
    # im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
    # cnt = contours[0]
    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(img,[box],0,(0,255,0),1)
    # cv2.imwrite(ruta_guardar+"img_minAreaRect_"+ str(numero_fruto) + version + name_file, img)

    # ##
    # img = img_copy
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # im2,contours,hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # # find the main island (biggest area)
    # cnt = contours[0]
    # max_area = cv2.contourArea(cnt)

    # for cont in contours:
    #     if cv2.contourArea(cont) > max_area:
    #         cnt = cont
    #         max_area = cv2.contourArea(cont)

    # # define main island contour approx. and hull
    # perimeter = cv2.arcLength(cnt,True)
    # epsilon = 0.1*cv2.arcLength(cnt,True)
    # approx = cv2.approxPolyDP(cnt,epsilon,True)
    # # print(approx)
    # hull = cv2.convexHull(cnt)
    # img = img_copy
    # cv2.drawContours(img, approx, -1, (0,255,0), 1)
    # cv2.imwrite(ruta_guardar+"img_approxPolyDP_"+ str(numero_fruto) + version + name_file, img)

    w_ = max(w_elipse,w_hough,w_enclosing)
    h_ = max(h_elipse,h_hough,h_enclosing)
    radius = max(radius_elipse,radius_hough,radius_enclosing) 
    angulo = angulo_elipse
    return w_, h_, radius, angulo