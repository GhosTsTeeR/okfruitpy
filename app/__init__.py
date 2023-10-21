from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
import mysql.connector
import binascii
from io import BytesIO
import fitz  

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
  obtener_pdf = 'guardar_analisis/pdf{}{}{}.pdf'.format(numero_aleatorio, version, fecha_actual)
  with open(obtener_pdf, 'rb') as pdf_file:
    pdf_binary = pdf_file.read()
    
  pdf_base64 = binascii.b2a_base64(pdf_binary).decode('utf-8')

  
  nombre_unico = numero_aleatorio+version+fecha_actual
  
  
  try:
    cnx = get_db_connection()  
    cursor = cnx.cursor()
    query = 'INSERT INTO Reporte ( nombreArchivo, datos) VALUES (%s, %s)'
    datos = (nombre_unico, pdf_base64)
    
    with cnx.cursor() as cursor:
      cursor.execute(query, datos)
      cnx.commit()

  except mysql.connector.Error as err:
    print(f"Error al insertar en MySQL: {err}")

  finally:
    cursor.close()
    cnx.close()
  nombr_unico = numero_aleatorio+version+fecha_actual
  cnx = get_db_connection()  
  cursor = cnx.cursor()
  query_uno = 'INSERT INTO Analisis_Fruta (idDocumento, descripcion, Tipo_Fruta_idTipo_Fruta, Huerto_idHuerto) VALUES (%s, %s, %s, %s)'
  datos_uno = (nombr_unico, "Descripcion relativa enviada desde la bd", 1, 1)
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
    datos_dos = (nombr_unico, numero_fruto, color, danio, pedicelo, calibre)
    #posiblemente haya fallos por la falta de calibre
    cursor.execute(query_dos, datos_dos)
    cnx.commit()
    cursor.close()
    cnx.close()
def add_datos_arandanos(data_json, numero_aleatorio, fecha_actual, version):
  obtener_pdf = 'guardar_analisis/pdf{}{}{}.pdf'.format(numero_aleatorio, version, fecha_actual)
  with open(obtener_pdf, 'rb') as pdf_file:
    pdf_binary = pdf_file.read()
    
  pdf_base64 = binascii.b2a_base64(pdf_binary).decode('utf-8')

  
  nombre_unico = numero_aleatorio+version+fecha_actual
  
  
  try:
    cnx = get_db_connection()  
    cursor = cnx.cursor()
    query = 'INSERT INTO Reporte ( nombreArchivo, datos) VALUES (%s, %s)'
    datos = (nombre_unico, pdf_base64)
    
    with cnx.cursor() as cursor:
      cursor.execute(query, datos)
      cnx.commit()

  except mysql.connector.Error as err:
    print(f"Error al insertar en MySQL: {err}")

  finally:
    cursor.close()
    cnx.close()
  
  
  cnx = get_db_connection()  
  cursor = cnx.cursor()
  query_uno = 'INSERT INTO Analisis_Fruta (idDocumento, descripcion, Tipo_Fruta_idTipo_Fruta, Huerto_idHuerto) VALUES (%s, %s, %s, %s)'
  datos_uno = (nombre_unico, "Descripcion relativa enviada desde la bd", 1, 1)
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
    datos_dos = (nombre_unico, numero_fruto, color, danio, calibre, manipulacion)
    #posiblemente haya fallos por la falta de calibre
    cursor.execute(query_dos, datos_dos)
    cnx.commit()
    cursor.close()
    cnx.close()
    
    
def export_pdf():
  cnx = get_db_connection()

  cursor = cnx.cursor()

  query = "SELECT datos FROM Reporte WHERE nombreArchivo='1253_v9_5_20231020'"

  cursor.execute(query)

  rows = cursor.fetchall()

  if rows is not None:
    binary_data = binascii.a2b_base64(rows[0][0])
    pdf_stream = BytesIO(binary_data)
    pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
    num_pages = pdf_document.page_count

    return pdf_stream
  else:
    return jsonify({"error": "No se encontraron datos en la base de datos"})
  




