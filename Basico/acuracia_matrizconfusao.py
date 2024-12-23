# Importando as bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Carregando o dataset Iris
data = load_iris()
X = data.data  # Recursos (características)
y = data.target  # Classes (alvos)

# Dividindo o dataset em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando um modelo simples (KNN)
model = KNeighborsClassifier(n_neighbors=3)  # KNN com 3 vizinhos
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calculando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")  # Exibe a acurácia com duas casas decimais

# Gerando a matriz de confusão - tn,fp,fpn,tp
#util para casos binarios
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# Exibindo a matriz de confusão de forma visual
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=data.target_names)
