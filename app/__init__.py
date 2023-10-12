from flask import Flask 
from flask_sqlalchemy import SQLAlchemy
import mysql.connector

db = SQLAlchemy()

def create_app():

  app = Flask(__name__)

  # Configuración para MySQL
  app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://admin:Okfruit#1@database-1.ca4e75aznuwa.us-east-2.rds.amazonaws.com'
  

  db.init_app(app)

  return app

def get_db_connection():
  cnx = mysql.connector.connect(user='admin', 
                                password='Okfruit#1',
                                host='database-1.ca4e75aznuwa.us-east-2.rds.amazonaws.com',
                                database='mydb')
  return cnx

def add_datos_cerezas(data_json, numero_aleatorio, fecha_actual, version):
  nombreUnico = numero_aleatorio+version+fecha_actual
  cnx = get_db_connection()  
  cursor = cnx.cursor()
  query_uno = 'INSERT INTO Analisis_Fruta (idDocumento, descripcion, Tipo_Fruta_idTipo_Fruta, Huerto_idHuerto) VALUES (%s, %s, %s, %s)'
  datos_uno = (nombreUnico, "Descripcion relativa enviada desde la bd", 1, 1)
  cursor.execute(query_uno, datos_uno)
  cnx.commit()
  cursor.close()
  cnx.close()
  for fruto in data_json['datos'][0]['resultadoAnalisis']:
    cnx = get_db_connection()  
    cursor = cnx.cursor()
    numero_fruto = fruto['NumeroFruto']
    color = fruto['Color']
    calibre = fruto['Calibre']
    pedicelo = fruto['Pedicelo']
    danio = fruto['Danio']

    query_dos = 'INSERT INTO Resultado_Cereza (idDocumento, Numero_Fruto, Color, Porcentaje_Daño, Pedicelo, calibre) VALUES (%s, %s, %s, %s, %s, %s)'
    datos_dos = (nombreUnico, numero_fruto, color, danio, pedicelo, calibre)
    #posiblemente haya fallos por la falta de calibre
    cursor.execute(query_dos, datos_dos)
    cnx.commit()
    cursor.close()
    cnx.close()
def add_datos_arandanos(data_json, numero_aleatorio, fecha_actual, version):
  print(data_json)
  nombreUnico = numero_aleatorio+version+fecha_actual
  cnx = get_db_connection()  
  cursor = cnx.cursor()
  query_uno = 'INSERT INTO Analisis_Fruta (idDocumento, descripcion, Tipo_Fruta_idTipo_Fruta, Huerto_idHuerto) VALUES (%s, %s, %s, %s)'
  datos_uno = (nombreUnico, "Descripcion relativa enviada desde la bd", 1, 1)
  cursor.execute(query_uno, datos_uno)
  cnx.commit()
  cursor.close()
  cnx.close()
  for fruto in data_json['datos'][0]['resultadoAnalisis']:
    cnx = get_db_connection()  
    cursor = cnx.cursor()
    numero_fruto = fruto['NumeroFruto']
    color = fruto['Color']
    calibre = fruto['Calibre']
    manipulacion = fruto['Manipulacion']
    danio = fruto['Danio']

    query_dos = 'INSERT INTO Resultado_Arandano (idDocumento, Numero_Fruto, Color, Porcentaje_Daño, calibre, manipulacion) VALUES (%s, %s, %s, %s, %s, %s)'
    datos_dos = (nombreUnico, numero_fruto, color, danio, calibre, manipulacion)
    #posiblemente haya fallos por la falta de calibre
    cursor.execute(query_dos, datos_dos)
    cnx.commit()
    cursor.close()
    cnx.close()


