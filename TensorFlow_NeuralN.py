import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 🔹 Datos de entrenamiento (Edad, Salario)
X = np.array([
    [22, 18000],
    [25, 25000],
    [47, 80000],
    [52, 150000],
    [23, 20000],
    [40, 72000],
    [35, 50000],
    [55, 180000]
])

# 🔹 Etiquetas: 0 = No compra, 1 = Compra
y = np.array([[0], [0], [1], [1], [0], [1], [0], [1]])

# Normalizar los datos (TensorFlow trabaja mejor con valores entre 0 y 1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 🔹 Definir el modelo de red neuronal en TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu", input_shape=(2,)),  # Capa oculta con 5 neuronas
    tf.keras.layers.Dense(1, activation="sigmoid")  # Capa de salida con 1 neurona (probabilidad)
])

# 🔹 Compilar el modelo
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 🔹 Entrenar la red neuronal
model.fit(X, y, epochs=100, verbose=1)

# 🔹 Prueba con un nuevo cliente (Edad: 30, Salario: 60,000)
nuevo_cliente = np.array([[30, 60000]])
nuevo_cliente = scaler.transform(nuevo_cliente)  # Normalizar

# Predicción
prediccion = model.predict(nuevo_cliente)
print(f"\nProbabilidad de compra del nuevo cliente: {prediccion[0][0]:.4f}")
