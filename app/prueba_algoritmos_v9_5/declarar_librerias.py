## librerias
import numpy as np
import cv2
import random
import argparse
import os
import os.path
import pandas as pd
import base64
import codecs, json
import glob
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import itertools
from PIL import Image
import math
import time

from skimage import color
from skimage import io
from skimage.util.dtype import dtype_range
from skimage import exposure
from skimage.color import rgb2lab, lab2lch

import xlsxwriter
import csv

# from sklearn import preprocessing
# from sklearn.preprocessing import StandardScaler

## Metodos
# from extraer_pixel import *
from prueba_algoritmos_v9_5.color import *## Colores de fruto definidos en LAB. 
# from square import square ## Reconoce el cuadrado del centro.
from prueba_algoritmos_v9_5.pedicelo import *
from prueba_algoritmos_v9_5.pudricion import * ## Reconoce pudricion, manchas cafes, ...
from prueba_algoritmos_v9_5.posicion import * 
from prueba_algoritmos_v9_5.calibre import *
from prueba_algoritmos_v9_5.calibre_2 import *
from prueba_algoritmos_v9_5.print_pdf import *
from prueba_algoritmos_v9_5.color_porcentajes import *


# import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
# import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle