# Importando as bibliotecas necessárias
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Carregando o dataset Iris como exemplo
data = load_iris()
X = data.data  # Features (atributos)
y = data.target  # Target (rótulos)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando um modelo KNeighborsClassifier
knn = KNeighborsClassifier()

# Definindo a grade de hiperparâmetros para ajuste
param_grid = {
    'n_neighbors': [3, 5, 7, 9],  # Número de vizinhos a considerar
    'weights': ['uniform', 'distance'],  # Tipo de peso para os vizinhos
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Métrica de distância
}

# Configurando o GridSearchCV
grid_search = GridSearchCV(
    estimator=knn,             # Modelo base
    param_grid=param_grid,     # Hiperparâmetros para testar
    cv=5,                      # Número de folds na validação cruzada
    scoring='accuracy',        # Métrica para avaliar os modelos
    verbose=2,                 # Nível de detalhes do log
    n_jobs=-1                  # Paralelismo para acelerar o processo
)

# Ajustando o GridSearchCV nos dados de treino
grid_search.fit(X_train, y_train)

# Exibindo os melhores parâmetros encontrados e o melhor desempenho
print("Melhores hiperparâmetros:", grid_search.best_params_)
print("Melhor precisão nos dados de validação:", grid_search.best_score_)

# Avaliando o melhor modelo nos dados de teste
best_knn = grid_search.best_estimator_  # Obtendo o modelo com os melhores parâmetros
test_score = best_knn.score(X_test, y_test)
print("Precisão nos dados de teste:", test_score)
