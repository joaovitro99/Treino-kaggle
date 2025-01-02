import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
# Exemplo de dataset
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [5, 4, 3, 2, 1],
    'Feature3': [1, 2, 2, 3, 4],
    'Target': [10, 9, 8, 7, 6]
})

# Calculando a correlação
correlation_matrix = data.corr()
print("Matriz de Correlação:")
print(correlation_matrix)

# Seleção com base no limiar de correlação (ex.: removendo variáveis com alta correlação entre si)
threshold = 0.9
correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            correlated_features.add(correlation_matrix.columns[i])

# Resultado após remover features correlacionadas
print("\nFeatures Correlacionadas Removidas:", correlated_features)
data_reduced = data.drop(columns=correlated_features)
print("\nDados após redução por correlação:")
print(data_reduced)

#---------

# Aplicando Variance Threshold
threshold = 0.1
selector = VarianceThreshold(threshold)
data_reduced = selector.fit_transform(data)

# Resultado após a seleção
print("\nFeatures com Variância Acima do Limiar:")
print(data.columns[selector.get_support()])

#---------
X = data[['Feature1', 'Feature2', 'Feature3']]
y = data['Target']

# Aplicando Chi-Square
chi_selector = SelectKBest(chi2, k=2)  # Seleciona as 2 melhores features
X_reduced = chi_selector.fit_transform(X, y)

# Resultado após a seleção
print("\nScores Chi-Square das Features:")
print(chi_selector.scores_)

print("\nDados após seleção com Chi-Square:")
print(pd.DataFrame(X_reduced, columns=X.columns[chi_selector.get_support()]))
