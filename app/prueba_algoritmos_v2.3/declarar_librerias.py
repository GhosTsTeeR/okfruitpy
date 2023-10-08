## librerias
import numpy as np
import cv2
from color_ import *
import argparse
from reportlab.pdfgen import canvas
# from square_3 import square
import pandas as pd
#import zbar
#scanner = zbar.Scanner()
import os

import math

import itertools
from reportlab.lib.pagesizes import A4

from array import *

from PIL import Image
import xlsxwriter
import glob
import json

## Metodos
# from preprocesamiento_v7 import limit_resolution
# from extraer_pixel import *
from color_ import * ## Colores de fruto definidos en LAB. 
from square_3 import square ## Reconoce el cuadrado del centro.
# from palito import *
# from pudricion import * ## Reconoce pudricion, manchas cafes, ...
# from posicion import * 
from print_pdf_v2_3 import *
from posicion import * 
from calibre_2 import *
from danio import *
from color_porcentajes import *

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


