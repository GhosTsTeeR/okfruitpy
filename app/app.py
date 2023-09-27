from flask_cors import CORS
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy 
from flask_marshmallow import Marshmallow
import pymysql
import __init__
from . import db

app = Flask(__name__)
CORS(app)

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

# consultar usuarios v:
@app.route('/productos', methods=['GET']) 
def get_usuario():
  all_usuarios = Usuario.query.all()
  result = usuario_schema.dump(all_usuarios)

  return jsonify(result)


if __name__ == "__main__":
    app.run(port=5000, debug=True)