# Importação de bibliotecas
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

# Carregando o dataset Iris como exemplo
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Inicializando o modelo base (Random Forest neste caso)
model = RandomForestClassifier()

# Aplicando Recursive Feature Elimination
rfe = RFE(estimator=model, n_features_to_select=2)  # Seleciona as 2 melhores features
X_reduced = rfe.fit_transform(X, y)

# Mostrando as features selecionadas
selected_features = X.columns[rfe.support_]
print("Features Selecionadas com RFE:")
print(selected_features)

# Resultado após a seleção
X_reduced_df = pd.DataFrame(X_reduced, columns=selected_features)
print("\nDados reduzidos:")
print(X_reduced_df)
