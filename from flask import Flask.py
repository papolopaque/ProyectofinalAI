from flask import Flask
import pandas as pd

app = Flask(__name__)

# Carga el CSV (ajusta el nombre y ruta si es necesario)
df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

@app.route('/')
def home():
    return "Aplicación funcionando"

@app.route('/salud')
def salud():
    return "OK", 200

if _name_ == "_main_":
    # Escuchar en todas las interfaces y puerto 5000 para Render
    app.run(host='0.0.0.0',port=5000)