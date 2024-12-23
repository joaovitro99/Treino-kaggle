# Importando bibliotecas necessárias
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Carregando o dataset Iris
data = load_iris()
X = data.data  # Recursos (características)
y = data.target  # Classes (alvos)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando um modelo de classificação (Random Forest)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Calculando as métricas

# Precisão: Mede a proporção de previsões corretas entre todas as previsões positivas.
# Fórmula: Precisão = Verdadeiros Positivos (TP) / (TP + Falsos Positivos (FP))

# Recall: Mede a proporção de exemplos positivos corretamente identificados.
# Fórmula: Recall = Verdadeiros Positivos (TP) / (TP + Falsos Negativos (FN))

# F1-Score: Combina precisão e recall, calculando a média harmônica entre eles.
# Fórmula: F1-Score = 2 * (Precisão * Recall) / (Precisão + Recall)

precision = precision_score(y_test, y_pred, average='macro')  # Macro = média entre classes
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Exibindo um relatório completo
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
