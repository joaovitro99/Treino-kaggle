# Importando bibliotecas necessárias
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform

# Carregando o dataset Iris como exemplo
data = load_iris()
X = data.data  # Features (atributos)
y = data.target  # Target (rótulos)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando um modelo KNeighborsClassifier
knn = KNeighborsClassifier()

# Definindo o espaço de hiperparâmetros para a busca aleatória
param_dist = {
    'n_neighbors': randint(1, 15),               # Número de vizinhos de 1 a 15
    'weights': ['uniform', 'distance'],          # Tipo de peso para os vizinhos
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Métrica de distância
}

# Configurando o RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=knn,              # Modelo base
    param_distributions=param_dist,  # Espaço de hiperparâmetros
    n_iter=30,                  # Número de combinações aleatórias a avaliar
    cv=5,                       # Número de folds na validação cruzada
    scoring='accuracy',         # Métrica para avaliar os modelos
    random_state=42,            # Garantir reprodutibilidade
    verbose=2,                  # Nível de detalhes do log
    n_jobs=-1                   # Paralelismo para acelerar o processo
)

# Ajustando o RandomizedSearchCV nos dados de treino
random_search.fit(X_train, y_train)

# Exibindo os melhores parâmetros e o melhor desempenho
print("Melhores hiperparâmetros encontrados:", random_search.best_params_)
print("Melhor precisão nos dados de validação:", random_search.best_score_)

# Avaliando o modelo com validação cruzada usando os melhores parâmetros
best_knn = random_search.best_estimator_  # Modelo com melhores parâmetros
cv_scores = cross_val_score(best_knn, X, y, cv=10, scoring='accuracy')  # Validação cruzada com 10 folds

# Exibindo os resultados da validação cruzada
print("Precisões na validação cruzada:", cv_scores)
print("Precisão média (cross-validation):", cv_scores.mean())

# Avaliando o modelo final nos dados de teste
test_score = best_knn.score(X_test, y_test)
print("Precisão no conjunto de teste:", test_score)
