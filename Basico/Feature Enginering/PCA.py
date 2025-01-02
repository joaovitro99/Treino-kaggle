# Importando bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Exemplo de dataset
data = pd.DataFrame({
    'Feature1': [2.5, 0.5, 2.2, 1.9, 3.1],
    'Feature2': [2.4, 0.7, 2.9, 2.2, 3.0],
    'Feature3': [1.5, 1.1, 1.6, 1.8, 1.3]
})

# 1. Padronizar os dados
# PCA é sensível à escala, então os dados precisam ser padronizados (média 0, desvio padrão 1)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 2. Aplicando PCA
# Reduzindo para 2 componentes principais
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# 3. Explicando a variância explicada
explained_variance = pca.explained_variance_ratio_
print(f"Variância explicada por cada componente: {explained_variance}")

# Resultado final
data_pca_df = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
print("\nDados após PCA:")
print(data_pca_df)
