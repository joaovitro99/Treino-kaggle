# Instalando scikit-optimize caso não esteja instalado
# !pip install scikit-optimize

# Importando bibliotecas necessárias
from skopt import BayesSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score

# Carregando o dataset Iris como exemplo
data = load_iris()
X = data.data  # Features (atributos)
y = data.target  # Target (rótulos)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando um modelo KNeighborsClassifier
knn = KNeighborsClassifier()

# Definindo o espaço de busca para a otimização bayesiana
param_space = {
    'n_neighbors': (1, 15),                     # Número de vizinhos de 1 a 15 (inteiro)
    'weights': ['uniform', 'distance'],         # Tipo de peso para os vizinhos
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Métrica de distância
}

# Configurando o BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=knn,                 # Modelo base
    search_spaces=param_space,     # Espaço de hiperparâmetros
    n_iter=30,                     # Número de iterações de busca
    cv=5,                          # Número de folds na validação cruzada
    scoring='accuracy',            # Métrica para avaliar os modelos
    random_state=42,               # Garantir reprodutibilidade
    verbose=2,                     # Nível de detalhes do log
    n_jobs=-1                      # Paralelismo para acelerar o processo
)

# Ajustando o BayesSearchCV nos dados de treino
bayes_search.fit(X_train, y_train)

# Exibindo os melhores parâmetros encontrados e o melhor desempenho
print("Melhores hiperparâmetros encontrados:", bayes_search.best_params_)
print("Melhor precisão nos dados de validação:", bayes_search.best_score_)

# Avaliando o modelo com validação cruzada usando os melhores parâmetros
best_knn = bayes_search.best_estimator_  # Modelo com melhores parâmetros
cv_scores = cross_val_score(best_knn, X, y, cv=10, scoring='accuracy')  # Validação cruzada com 10 folds

# Exibindo os resultados da validação cruzada
print("Precisões na validação cruzada:", cv_scores)
print("Precisão média (cross-validation):", cv_scores.mean())

# Avaliando o modelo final nos dados de teste
test_score = best_knn.score(X_test, y_test)
print("Precisão no conjunto de teste:", test_score)
