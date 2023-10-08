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

def add_datos(data_json):
  print(data_json)