from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy 
from flask_marshmallow import Marshmallow
import random
import datetime

db = SQLAlchemy()
from __init__ import get_db_connection, add_user, login_user
from prueba_algoritmos_v9_5.demo_main_v9_5 import proceso_analisis
from prueba_algoritmos_v2_3.demo_main_v2_3 import inicializar_arandanos

app = Flask(__name__)
CORS(app)

# HTPP
# inicializar sqlalchemy y marshmallow
ma = Marshmallow(app)

# consultar frutas v:
@app.route('/get_type_fruit')
def consultar_tipo_fruta():
  cnx = get_db_connection()

  cursor = cnx.cursor()

  query = "SELECT * FROM Tipo_Fruta;"

  cursor.execute(query)

  rows = cursor.fetchall()
  return jsonify(rows)

# Consultar UsuariosRegistrados Esta ruta para validar Login BASICA****
@app.route('/get_usuarios/<string:username>/<string:password>', methods=['GET'])
def obtenerUsuario(username, password):
    cnx = get_db_connection()
    cursor = cnx.cursor()
    query = "SELECT * FROM Usuario WHERE nombres = %s AND password = %s;"
    cursor.execute(query, (username, password))

    row = cursor.fetchone()  # Obtener la primera fila que cumple con los criterios de consulta
    print("----------------------------------------------------")

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
    print(request)
    tipo = request.json['selecction']
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

@app.route('/add-usuario', methods=['POST'])
def add_usuario():
  correo = request.json['email']
  password = request.json['password']
  proceso=add_user(correo, password)
  return jsonify({
  "code": 200, 
  "message": "Exito" 
})

#logueo Usuario
#logueo Usuario
@app.route('/logeo_user', methods=['POST'])
def logeo_user():
  correo = request.json.get('nombre')
  password = request.json.get('contrasena')

  if correo is None or password is None:
      return jsonify({"error": "Correo y contraseña son requeridos"}), 400

  status, user_email = login_user(correo, password)

  if status == 200:
      return jsonify({"message": "Inicio de sesión exitoso", "nombre": user_email})
  elif status == 401:
      return jsonify({"error": "Credenciales inválidas"}), 401
  else:
      return jsonify({"error": "Error en el servidor"}), 500


#************** Se define el HOST para poder acceder a la API
# en la red local de nuestro hogar

# host = **IP de nuestro dispositivo a nivel WIFI **
# Se ingresa a CMD, se ejecuta el comando -ipconfig /all-
# Se saca la IP que aparece en Dirección IPv4

# port = ** El puerto se puede dejar asi.


# Al ejecutar la API en CMD toca dar permisos de Administrador
# Ejecutar CMD como administrador.
if __name__ == "__main__":
    app.run(host='192.168.1.7',port=5000)