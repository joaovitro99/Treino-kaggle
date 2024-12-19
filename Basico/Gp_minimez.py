# Importando bibliotecas necessárias
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np

# Carregando o dataset Iris
data = load_iris()
X = data.data
y = data.target

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo o espaço de busca
space = [
    Integer(1, 15, name="n_neighbors"),                     # Número de vizinhos (1 a 15)
    Categorical(['uniform', 'distance'], name="weights"),   # Pesos dos vizinhos
    Categorical(['euclidean', 'manhattan', 'minkowski'], name="metric")  # Métrica de distância
]

# Definindo a função objetivo
def objective(params):
    n_neighbors, weights, metric = params  # Extraindo os parâmetros do vetor
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )
    # Avaliação do modelo com validação cruzada
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return -np.mean(scores)  # Retornamos o negativo da acurácia média para minimizar

# Executando o gp_minimize
result = gp_minimize(
    func=objective,   # Função objetivo a ser minimizada
    dimensions=space, # Espaço de busca
    n_calls=30,       # Número de avaliações
    random_state=42   # Para resultados reprodutíveis
)

# Exibindo os melhores parâmetros encontrados
best_params = result.x
print("Melhores parâmetros encontrados:")
print(f"n_neighbors: {best_params[0]}, weights: {best_params[1]}, metric: {best_params[2]}")

# Avaliando o modelo com os melhores parâmetros
best_model = KNeighborsClassifier(
    n_neighbors=best_params[0],
    weights=best_params[1],
    metric=best_params[2]
)
cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='accuracy')
print("Precisões na validação cruzada:", cv_scores)
print("Precisão média (cross-validation):", cv_scores.mean())

# Avaliação no conjunto de teste
best_model.fit(X_train, y_train)
test_score = best_model.score(X_test, y_test)
print("Precisão no conjunto de teste:", test_score)
