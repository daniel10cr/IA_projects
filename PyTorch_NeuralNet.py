import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Datos de ejemplo: Edad (input) y Salario (output)
edad = np.array([[22], [25], [30], [35], [40], [45], [50], [55], [60]], dtype=np.float32)
salario = np.array([[25000], [28000], [35000], [42000], [50000], [58000], [65000], [70000], [75000]], dtype=np.float32)

# Convertimos los datos a tensores de PyTorch
X = torch.tensor(edad, dtype=torch.float32)
y = torch.tensor(salario, dtype=torch.float32)

# Definir la red neuronal
class RedNeuronal(nn.Module):
    def __init__(self):
        super(RedNeuronal, self).__init__()
        self.capa_oculta = nn.Linear(1, 10)  # Capa oculta con 10 neuronas
        self.activacion = nn.ReLU()  # Función de activación ReLU
        self.salida = nn.Linear(10, 1)  # Capa de salida

    def forward(self, x):
        x = self.capa_oculta(x)
        x = self.activacion(x)
        x = self.salida(x)
        return x

# Crear el modelo
modelo = RedNeuronal()

# Definir la función de pérdida y el optimizador
criterio = nn.MSELoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.01)

# Entrenamiento del modelo
epochs = 1000
for epoch in range(epochs):
    optimizador.zero_grad()  # Reiniciar gradientes
    predicciones = modelo(X)  # Hacer predicción
    perdida = criterio(predicciones, y)  # Calcular pérdida
    perdida.backward()  # Retropropagación
    optimizador.step()  # Actualizar pesos

    if epoch % 100 == 0:
        print(f"Época {epoch}, Pérdida: {perdida.item()}")

# Hacer una predicción
edad_nueva = torch.tensor([[27]], dtype=torch.float32)
salario_predicho = modelo(edad_nueva).item()
print(f"\nPredicción para una persona de 27 años: ${salario_predicho:.2f}")
