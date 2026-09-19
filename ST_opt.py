from sentence_transformers import SentenceTransformer, InputExample, losses, SentencesDataset, LoggingHandler
from torch.utils.data import DataLoader

# biblioteca de optimización de hiperparámetros para un ajuste eficiente del modelo de aprendizaje profundo
import optuna 
import torch

# ---------------------------------------------------
# Datos de entrenamiento y validación
# Simula pares de oraciones con etiquetas de similitud entre 0 y 1 "label=0.8"
# ---------------------------------------------------

# Ejemplos para enseñar o entrenar al modelo
train_samples = [
    InputExample(texts=["Zapatos para correr", "zapatos deportivos para correr bajo la lluvia"], label=0.8),
    InputExample(texts=["Me gusta correr por las mañanas", "Odio correr"], label=0.2),
    InputExample(texts=["Tenis deportivos para entrenamientos", "Botas elegantes para la oficina"], label=0.4), 
    InputExample(texts=["Ropa deportiva para mujeres", "Sudadera entrenamiento femenina"], label=0.8), 
    ]

# Ejemplos para evaluar qué tan bien generaliza
val_samples = [
    InputExample(texts=["Salir a correr es divertido", "Me gusta correr por la mañana"], label=0.9),
    InputExample(texts=["Me gusta practicar deportes", "Adoro jugar fútbol"], label=0.7), 
    InputExample(texts=["No me gustan los deportes", "Me gusta hacer ejercicio"], label=0.1),
    ]

# ---------------------------------------------------
# definimos una Función objetivo para Optuna (que prueba muchas configuraciones posibles):
# learning_rate, tamaño de batch y número de épocas; vector de configuración para mejor rendimiento
# ---------------------------------------------------
def objective(trial):

    # Hiperparámetros que queremos optimizar
    
    # tasa de aprendizaje dentro de Optuna
    lr = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    # controla el número de oraciones procesadas simultáneamente durante el entrenamiento
    train_batch = trial.suggest_int("batch_size", 4, 32)
    # cuántas veces el modelo procesa todo el conjunto de datos de entrenamiento
    epochs = trial.suggest_int("epochs", 1, 3)

    # creamos y cargamos el modelo
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # SentencesDataset clase de la biblioteca SentenceTransformer para "envolver" una lista de objetos
    train_dataset = SentencesDataset(train_samples, model)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch)

    # Usamos CosineSimilarityLoss en la búsqueda semántica
    train_loss = losses.CosineSimilarityLoss(model)

    # Entrenamos el modelo utilizando los datos "train_loader" y "train_loss" 
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=10,
        optimizer_params={"lr": lr} )

    # -----------------------
    # Evaluamos el modelo. Comparamos la similitud calculada con la etiqueta real
    # -----------------------
    error_total = 0
    for sample in val_samples: # frases (1 y 2) de val_samples([frase1, frase2],label)
        emb1, emb2 = model.encode(sample.texts) # el modelo convierte cada frase en su embedding,
        cos_sim = torch.nn.functional.cosine_similarity( 
            torch.tensor(emb1), torch.tensor(emb2), dim=0 # mide qué tan similares son ambos vectores
        ).item()

        # Error = diferencia con la etiqueta esperada
        error_total += abs(cos_sim - sample.label)

    error_medio = error_total / len(val_samples)

    return error_medio


# ---------------------------------------------------
# Corremos el estudio de optimización usando Optuna, definiendo el objeto study de la biblioteca
# optuna con el método "create study" y llamando a la función "objective" con 10 intentos
# ---------------------------------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# imprimimos los resultados del estudio para que muestre los parámetros "más" óptimos
print("\n")
print("\nMejores hiperparámetros encontrados:")
print(study.best_params)
print("\n")
print("\n")