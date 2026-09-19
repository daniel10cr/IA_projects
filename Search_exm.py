from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity #sklearn - biblioteca muy usada en machine learning

# 'all-MiniLM-L6-v2' modelo preentrenado (en Sentence Transformer) para convertir texto en vectores
# Está basado en una arquitectura de Transformer llamada MiniLM
model = SentenceTransformer('all-MiniLM-L6-v2')

# Documentos a buscar
documents = [
    "Zapatos para correr",
    "Zapatos impermeables para clima lluvioso",
    "Tenis deportivos para entrenamientos",
    "Botas elegantes para la oficina",
    "Ropa deportiva para mujeres"
]

# Consulta del usuario
query = "zapatos deportivos para correr bajo la lluvia"

# Convertimos documentos y la consulta en vectores
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

# Calculamos la similitud entre la consulta y los documentos
# El [0] al final se usa porque cosine_similarity devuelve una matriz, 
# y con eso nos quedamos con la fila que nos interesa (la única de la consulta).
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

# Mostramos los resultados ordenados por similitud
results = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)

print("Resultados más relevantes:")
for doc, score in results:
    print(f"{doc} (similitud: {score:.2f})")
