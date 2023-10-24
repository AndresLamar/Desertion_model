import numpy as np
import joblib
from flask import Flask,render_template,request


app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Cargar el modelo al iniciar la aplicación
modelo_cargado = joblib.load('modelo_arbol_decision.pkl')
print("Modelo cargado exitosamente:", modelo_cargado)

    
@app.route('/prediccion', methods=['GET','POST'])   
def predict():
    if request.method =='POST':        
        try:
            edad = float(request.form['var_1'])
            genero = float(request.form['var_2'])
            estrato = float(request.form['var_3'])
            materias_perdidas = float(request.form['var_4'])
            ultimo_promedio_academico = float(request.form['var_5'])

            # # Crear un array con los datos
            datos_prueba = np.array([[edad, genero, estrato, materias_perdidas, ultimo_promedio_academico]])

            # se hace una predicción
            model_prediction=modelo_cargado.predict(datos_prueba)
            model_prediction=round(float(model_prediction),2)
           
            # se muestra el resultado
            if (model_prediction == 1):
                res = "Deserta"
            else:
                res = "No deserta"
            
        except ValueError:
             return "Por favor ingresa valores correctos"
        return render_template('prediccion.html',prediccion=res)
    
    if __name__=='__main__':
     app.run(debug=True, use_reloader=False)

    
if __name__=='__main__':
     app.run(debug=True)

