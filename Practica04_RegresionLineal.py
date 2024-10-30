import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

datos = pd.read_csv('datos04.csv')

# Variables independientes (X) y dependiente (Y)
x = datos[['Horas_de_estudio', 'Horas_de_sueno', 'Participacion']].values
y = datos['Calificacion'].values

# Entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(x, y)

# Promedios de variables
promedio_estudio = datos['Horas_de_estudio'].mean()
promedio_sueño = datos['Horas_de_sueno'].mean()
promedio_participacion = datos['Participacion'].mean()

# Datos de entrada
horas_estudio = [3, 5, 7]
horas_sueño = [4, 6, 8]
participacion = [5, 7, 9]

# Predicciones para horas de estudio
predicciones_estudio = []
for e in horas_estudio:
    prediccion = modelo.predict([[e, promedio_sueño, promedio_participacion]])
    print(f"Para {e} horas de estudio la predicción es: {prediccion[0]:.4f}")
    predicciones_estudio.append(prediccion[0])

# Predicciones para horas de sueño
predicciones_sueño = []
for s in horas_sueño:
    prediccion = modelo.predict([[promedio_estudio, s, promedio_participacion]])
    print(f"Para {s} horas de sueño la predicción es: {prediccion[0]:.4f}")
    predicciones_sueño.append(prediccion[0])

# Predicciones para participación
predicciones_participacion = []
for p in participacion:
    prediccion = modelo.predict([[promedio_estudio, promedio_sueño, p]])
    print(f"Para {p} de participación la predicción es: {prediccion[0]:.4f}")
    predicciones_participacion.append(prediccion[0])

# Evaluacion del modelo
r2 = modelo.score(x, y)
print(f"Coeficiente de determinacion R2: {r2:.4f}")

# Obtener el mse
Y_pred = modelo.predict(x)
mse = mean_squared_error(y, Y_pred)
print(f"Error cuadratico medio (MSE): {mse:.4f}")

# Datos ordenados de horas de estudio
horas_estudio_max = datos['Horas_de_estudio'].max()
horas_estudio_min = datos['Horas_de_estudio'].min()

x_estudio = []
y_estudio = []

for e in range(int(horas_estudio_min), int(horas_estudio_max) + 1):
    prediccion = modelo.predict([[e, promedio_sueño, promedio_participacion]])
    x_estudio.append(e)
    y_estudio.append(prediccion[0])

# Datos ordenados de horas de sueño
horas_sueño_max = datos['Horas_de_sueno'].max()
horas_sueño_min = datos['Horas_de_sueno'].min()

x_sueño = []
y_sueño = []

for s in range(int(horas_sueño_min), int(horas_sueño_max) + 1):
    prediccion = modelo.predict([[promedio_estudio, s, promedio_participacion]])
    x_sueño.append(s)
    y_sueño.append(prediccion[0])

# Datos ordenados de participacion
participacion_max = datos['Participacion'].max()
participacion_min = datos['Participacion'].min()

x_participacion = []
y_participacion = []

for p in range(int(participacion_min), int(participacion_max) + 1):
    prediccion = modelo.predict([[promedio_estudio, promedio_sueño, p]])
    x_participacion.append(p)
    y_participacion.append(prediccion[0])


plt.figure(figsize=(10, 6))

# Lineas se regresion
plt.plot(x_estudio, y_estudio, color='blue', label='Calificación vs. Horas de Estudio')
plt.plot(x_sueño, y_sueño, color='orange', label='Calificación vs. Horas de Sueño')
plt.plot(x_participacion, y_participacion, color='green', label='Calificación vs. Participacion')

# Valores reales
plt.scatter(datos['Horas_de_estudio'], y, color='gray',label='Datos reales')

# Predicciones
plt.scatter(horas_estudio, predicciones_estudio, color='red', marker='x',label='Predicciones (Horas de estudio)')
plt.scatter(horas_sueño, predicciones_sueño, color='purple', marker='x', label='Predicciones (Horas de Sueño)')
plt.scatter(participacion, predicciones_participacion, color='orange', marker='x',label='Predicciones (Participaciones)')

plt.xlabel('Variables independientes')
plt.ylabel('Calificación')
plt.title('Calificación en funcion de Horas de estudio, Horas de Sueño y Participación')
plt.legend()
plt.grid()
plt.show()
