from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy 
from flask_marshmallow import Marshmallow
import random
import datetime

db = SQLAlchemy()
from __init__ import get_db_connection, export_pdf
from prueba_algoritmos_v9_5.demo_main_v9_5 import proceso_analisis
from prueba_algoritmos_v2_3.demo_main_v2_3 import inicializar_arandanos


app = Flask(__name__)
CORS(app)

# HTPP
# inicializar sqlalchemy y marshmallow
ma = Marshmallow(app)

# modelo de tabla 
class Usuario(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  nombre = db.Column(db.String(100), unique=True)
  correo = db.Column(db.String(200))
  pssw = db.Column(db.String(200))

  def __init__(self, nombre, correo, pssw):
    self.nombre = nombre
    self.correo = correo 
    self.pssw = pssw

# schema del usuario
class UsuarioSchema(ma.Schema):
  class Meta:
    fields = ('id', 'nombre', 'correo', 'pssw')

usuario_schema = UsuarioSchema()
pusuarios_schema = UsuarioSchema(many=True)

# crear un nuevo usuario
@app.route('/usuario', methods=['POST'])
def add_usuario():
  nombre = request.json['nombre']
  correo = request.json['correo']
  pssw = request.json['pssw']

  nuevo_usuario = Usuario(nombre, correo, pssw)

  db.session.add(nuevo_usuario)
  db.session.commit()

  return usuario_schema.jsonify(nuevo_usuario)

# consultar frutas v:
@app.route('/get_type_fruit')
def consultar_tipo_fruta():
  cnx = get_db_connection()

  cursor = cnx.cursor()

  query = "SELECT * FROM Tipo_Fruta;"

  cursor.execute(query)

  rows = cursor.fetchall()
  return jsonify(rows)

# Consultar UsuariosRegistrados
@app.route('/get_usuarios/<string:username>/<string:password>', methods=['GET'])
def obtenerUsuario(username, password):
    cnx = get_db_connection()
    cursor = cnx.cursor()
    query = "SELECT * FROM Usuario WHERE nombres = %s AND password = %s;"
    cursor.execute(query, (username, password))

    row = cursor.fetchone()  # Obtener la primera fila que cumple con los criterios de consulta
    print(row)

    if row is not None:
        return jsonify(row)
    else:
        return jsonify({"mensaje": "Usuario no encontrado"})


@app.route('/post_type_fruit', methods=['POST'])  
def insert_type_fruit():

  cnx = get_db_connection()  
  cursor = cnx.cursor()

  tipo = request.json['tipo']
  descripcion = request.json['descripcion']  

  query = "INSERT INTO Tipo_Fruta (tipo, descripcion) VALUES (%s, %s)"
  datos = (tipo, descripcion)

  cursor.execute(query, datos)

  cnx.commit()  

  cursor.close()
  cnx.close()

  return "tipo de fruta insertado insertado!"
@app.route('/analisis', methods=['POST'])
def insert_analisis_img():
    #tipo = request.json['selecction']
    tipo = "arandanos"
    print(tipo)
    # Verificar si se recibió un archivo en la solicitud POST
    if 'file' not in request.files:
        return "No se ha proporcionado un archivo."

    # Obtener el archivo de la solicitud POST
    file = request.files['file']
    if (tipo == "cerezas"):
      # Verificar si el archivo tiene un nombre y es una imagen
      if file.filename == '':
          return "El archivo no tiene un nombre válido."
      if not file.content_type.startswith('image/'):
          return "El archivo no es una imagen válida."

      ruta_img = "image/"
      fecha_actual = datetime.datetime.now().strftime("%Y%m%d")
      numero_aleatorio = str(random.randint(1000, 9999))

      nombre_archivo = "img" + numero_aleatorio + "_" + fecha_actual + ".jpg"

      file.save(ruta_img + nombre_archivo)

      proceso_analisis(ruta_img, nombre_archivo, numero_aleatorio, fecha_actual)
      return "Se verificó el archivo y se generó un reporte con éxito!"
    elif (tipo == "arandanos"):
      # Verificar si el archivo tiene un nombre y es una imagen
      if file.filename == '':
          return "El archivo no tiene un nombre válido."
      if not file.content_type.startswith('image/'):
          return "El archivo no es una imagen válida."

      ruta_img = "image/"
      fecha_actual = datetime.datetime.now().strftime("%Y%m%d")
      numero_aleatorio = str(random.randint(1000, 9999))

      nombre_archivo = "img" + numero_aleatorio + "_" + fecha_actual + ".jpg"

      file.save(ruta_img + nombre_archivo)

      inicializar_arandanos(ruta_img, nombre_archivo, numero_aleatorio, fecha_actual)
      return "Se verificó el archivo y se generó un reporte con éxito!"
    else:
      print("el nombre no coincide ni con arandanos ni con cerezas")
      

    return "Hubo un problema!"

@app.route('/descargar_pdf', methods=['GET'])
def descargar_pdf():
  
  pdf= export_pdf()
  response = send_file(
        pdf,
        as_attachment=True,  # Esto hará que el navegador ofrezca descargar el archivo
        download_name='hola.pdf'
    )

  return response;



if __name__ == "__main__":
    app.run(host='192.168.94.98',port=5000)