import random as rd 

resultados = []

for i in range(1000):

    dado1 = rd.randint(1,6)
    dado2 = rd.randint(1,6)

    suma = dado1 + dado2

    resultados.append(suma)

# Histograma
import matplotlib.pyplot as plt


plt.hist(resultados, bins=38, color='blue', edgecolor='black')
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.show()