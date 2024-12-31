# Importando bibliotecas necessárias
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Gerando um exemplo de dados (uma matriz com 5 amostras e 2 características)
data = np.array([[1, 200], [2, 300], [3, 400], [4, 500], [5, 600]])

# 1. Padronização (Standardization)
# A padronização transforma os dados para que tenham média 0 e desvio padrão 1.
# Fórmula: z = (x - média) / desvio padrão
scaler_standard = StandardScaler()
data_standardized = scaler_standard.fit_transform(data)
print("Dados Padronizados (Standardization):")
print(data_standardized)

# 2. Normalização (Normalization)
# A normalização escala os valores para um intervalo específico, geralmente [0, 1].
# Fórmula: x' = (x - min(x)) / (max(x) - min(x))
scaler_minmax = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler_minmax.fit_transform(data)
print("\nDados Normalizados (Normalization):")
print(data_normalized)
