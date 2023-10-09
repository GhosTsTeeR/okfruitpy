# # import pandas as pd
# from sklearn import svm
# from sklearn.model_selection import GridSearchCV
# # import os
# import matplotlib.pyplot as plt
# from skimage.transform import resize
# from skimage.io import imread
# # import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
# import pickle

from prueba_algoritmos_v9_5.declarar_librerias import *
from prueba_algoritmos_v9_5.declarar_variables import *
from prueba_algoritmos_v9_5.declarar_rutas import *

##
datadir = r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC/imágenes/24-04-2022 (Cerezas individualizadas)'

name_file = datadir.split("/")[-1]
name_folder = datadir.split("/")[-2]

## Ruta para guardar resultados
ruta_guardar = os.path.join(r'/Users/sergiobaltierra/OneDrive - Universidad Autónoma de Chile/Proyectos/FIC/resultados/' + name_file + version + '/')
if not os.path.exists(ruta_guardar):
    os.makedirs(ruta_guardar)

# Categories=['Fruta sana','Fruta dañada']
# print("Type y to give categories or type n to go with classification of Fruta sana and Fruta dañada")

# while(True):
#   check=input()
#   if(check=='n' or check=='y'):
#     break
#   print("Please give a valid input (y/n)")
# if(check=='y'):
# print("Enter How Many types of Images do you want to classify")
# n=int(input())
# Categories=[]
# print(f'please enter {n} names')
# for i in range(n):
#   name=input()
#   Categories.append(name)
# print(f"If not drive Please upload all the {n} category images in google collab with the same names as given in categories")

flat_data_arr=[]
target_arr=[]
#please use datadir='/content' if the files are upload on to google collab
#else mount the drive and give path of the parent-folder containing all category images folders.
# datadir='/content/drive/MyDrive/ML'

# Categories=[]
# Categories.append("Fruta sana")
# Categories.append("Fruta dañada")

for i in Categories:
  print(f'loading... category : {i}')
  path=os.path.join(datadir,i)
  for img in os.listdir(path):
    # print(os.path.join(path,img))
    if(os.path.join(path,img).split("/")[-1] != ".DS_Store"):
      img_array=imread(os.path.join(path,img))
      img_resized=resize(img_array,(150,150,3))
      flat_data_arr.append(img_resized.flatten())
      target_arr.append(Categories.index(i))
  print(f'loaded category:{i} successfully')
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data)
df['Target']=target

x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
print('Splitted Successfully')

param_grid = {'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.01,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
print("The training of the model is started, please wait for while as it may take few minutes to complete")
model=GridSearchCV(svc,param_grid)
model.fit(x_train,y_train)
print('The Model is trained well with the given images')
model.best_params_

y_pred=model.predict(x_test)
print("The predicted Data is :")
y_pred

print("The actual data is:")
np.array(y_test)

#classification_report(y_pred,y_test)
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
#confusion_matrix(y_pred,y_test)

pickle.dump(model,open(ruta_guardar + 'img_model.p','wb'))
print("Pickle is dumped successfully")



