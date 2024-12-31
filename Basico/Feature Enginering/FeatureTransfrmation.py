import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import zscore

# Exemplo de DataFrame
data = pd.DataFrame({
    'Age': [25, np.nan, 35, 45, 50, np.nan, 65],
    'Gender': ['Male', 'Female', np.nan, 'Female', 'Male', 'Male', 'Female'],
    'Income': [3000, 4000, 5000, 6000, 7000, 8000, 100000]  # Contém outlier
})

# 1. Lidando com Dados Faltantes
# Substituindo valores NaN por média (para numéricos) e pela moda (para categóricos).
imputer_mean = SimpleImputer(strategy='mean')
data['Age'] = imputer_mean.fit_transform(data[['Age']])  # Imputação numérica
imputer_mode = SimpleImputer(strategy='most_frequent')
data['Gender'] = imputer_mode.fit_transform(data[['Gender']])  # Imputação categórica

# 2. Transformando Dados Categóricos
# Convertendo a coluna 'Gender' para variáveis dummy (One-Hot Encoding).
encoder = OneHotEncoder(sparse=False, drop='first')  # Drop para evitar multicolinearidade
gender_encoded = encoder.fit_transform(data[['Gender']])
data_encoded = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out(['Gender']))
data = pd.concat([data.drop(columns='Gender'), data_encoded], axis=1)

# 3. Detecção de Outliers
# Usando o Z-Score: valores com Z acima de 3 ou abaixo de -3 são considerados outliers.
z_scores = zscore(data['Income'])
data['IsOutlier'] = np.abs(z_scores) > 3  # Flag para outliers

data = data[data['IsOutlier'] == False].drop(columns='IsOutlier')

# Resultado final
print("Dados Finalizados:")
print(data)
