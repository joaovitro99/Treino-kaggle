from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# **Divisão de dados - `train_test_split`**
# Divide os dados em conjuntos de treino e teste
X = [[1], [2], [3], [4], [5]]
y = [1, 2, 3, 4, 5]

# **Divisão de dados - `train_test_split`**
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)  # Dados simulados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Regressão Linear**
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Erro Quadrático Médio (Linear Regression): {mean_squared_error(y_test, y_pred)}")

# **Gradient Boosting Regressor**
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print(f"Erro Quadrático Médio (Gradient Boosting): {mean_squared_error(y_test, y_pred_gb)}")

# **Classificação com K-Nearest Neighbors (KNN)**
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)  # KNN com 3 vizinhos
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(f"Acurácia (KNN): {accuracy_score(y_test, y_pred_knn)}")

# **Random Forest Classifier**
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred_rf = clf.predict(X_test)
print(f"Acurácia (Random Forest): {accuracy_score(y_test, y_pred_rf)}")

# **Clustering com K-Means**
data = [[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]
kmeans = KMeans(n_clusters=2, random_state=42)  # Agrupando em 2 clusters
kmeans.fit(data)
print(f"Centroides dos clusters: {kmeans.cluster_centers_}")
print(f"Rótulos atribuídos: {kmeans.labels_}")

# **Padronização com `StandardScaler`**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# **Métricas para Regressão e Classificação**
# - Regressão: `mean_squared_error` para medir o erro
# - Classificação: `accuracy_score` para medir a precisão
