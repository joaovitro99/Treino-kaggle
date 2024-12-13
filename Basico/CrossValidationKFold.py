# Importando o KFold para criar divisões de treino e teste
from sklearn.model_selection import KFold

# Configurando o KFold para dividir os dados em 3 partes (splits)
kf = KFold(n_splits=3)

# Visualizando o objeto KFold (ele define como os dados serão divididos)
print(kf)

# Iterando sobre as divisões (splits) para obter índices de treino e teste
for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print(train_index, test_index)  # Exibe os índices dos dados de treino e teste em cada split

# Função para treinar um modelo e retornar o escore no conjunto de teste
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)  # Treina o modelo com os dados de treino
    return model.score(X_test, y_test)  # Calcula o escore no conjunto de teste

# Importando o StratifiedKFold para validação estratificada
from sklearn.model_selection import StratifiedKFold

# Configurando o StratifiedKFold para dividir os dados em 3 partes estratificadas
folds = StratifiedKFold(n_splits=3)

# Listas para armazenar os escores de diferentes modelos
scores_logistic = []
scores_svm = []
scores_rf = []

# Realizando a validação cruzada estratificada com os dados do conjunto digits
for train_index, test_index in folds.split(digits.data, digits.target):
    # Dividindo os dados em treino e teste com base nos índices gerados pelo StratifiedKFold
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    
    # Calculando o escore do modelo de regressão logística
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear', multi_class='ovr'), 
                                     X_train, X_test, y_train, y_test))
    
    # Calculando o escore do modelo SVM
    scores_svm.append(get_score(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    
    # Calculando o escore do modelo Random Forest
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

# Exibindo os escores calculados para cada modelo
print(scores_logistic)
print(scores_svm)
print(scores_rf)

# Importando a função cross_val_score para simplificar a validação cruzada -- Bem mais simples
from sklearn.model_selection import cross_val_score

# Validando o modelo de regressão logística com validação cruzada
cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), digits.data, digits.target, cv=3)

# Validando o modelo SVM com validação cruzada
cross_val_score(SVC(gamma='auto'), digits.data, digits.target, cv=3)

# Validando o modelo Random Forest com validação cruzada
cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target, cv=3)

# Ajustando o número de árvores na Random Forest para encontrar o melhor desempenho
# Usando validação cruzada para avaliar o desempenho com diferentes números de estimadores

# Random Forest com 5 árvores
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5), digits.data, digits.target, cv=10)
print(np.average(scores1))  # Calcula a média dos escores

# Random Forest com 20 árvores
scores2 = cross_val_score(RandomForestClassifier(n_estimators=20), digits.data, digits.target, cv=10)
print(np.average(scores2))  # Calcula a média dos escores
