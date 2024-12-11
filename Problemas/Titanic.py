import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from google.colab import files

# Carregamento dos dados
treino = pd.read_csv('train.csv')
teste = pd.read_csv('test.csv')
respostas = pd.read_csv('gender_submission.csv')

def gerar_csv_previsao(teste, predictions, nome_arquivo='submission.csv'):
    # Verificar se o teste contém a coluna 'PassengerId'
    if 'PassengerId' not in teste.columns:
        raise ValueError("O DataFrame de teste não contém a coluna 'PassengerId'.")

    # Criar um DataFrame para as previsões
    submission = pd.DataFrame({
        'PassengerId': teste['PassengerId'],
        'Survived': predictions
    })

    # Ordenar por PassengerId (opcional, mas pode ser uma boa prática)
    submission.sort_values(by='PassengerId', inplace=True)

    # Garantir que o DataFrame tenha exatamente 418 entradas
    if len(submission) != 418:
        raise ValueError("O DataFrame de submissão deve ter exatamente 418 entradas.")

    # Salvar como arquivo CSV
    submission.to_csv(nome_arquivo, index=False)

    print(f"Arquivo {nome_arquivo} gerado com sucesso!")

def normaliza_numeros(df):
    colunas_numericas = df.select_dtypes(include=[np.number]).columns

    if len(colunas_numericas) == 0:
        return df  # Retorna o DataFrame sem alterações se não houver colunas numéricas

    # Normalização Min-Max
    min_vals = df[colunas_numericas].min()
    max_vals = df[colunas_numericas].max()

    # Evitar divisão por zero
    data_normalizado = (df[colunas_numericas] - min_vals) / (max_vals - min_vals)

    # Substitui colunas no DataFrame original
    df[colunas_numericas] = data_normalizado

    return df

def normaliza_palavra(df):
    df['Sex'] = df['Sex'].str.lower().str.strip().str.replace('[^\w\s]', '', regex=True)
    df['Embarked'] = df['Embarked'].str.lower().str.strip().str.replace('[^\w\s]', '', regex=True)

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    return df

def bota_faltantes(df):
    # Preenchendo valores numéricos com a média
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    for col in colunas_numericas:
        df[col] = df[col].fillna(df[col].mean())  # Preenchendo com média diretamente

    # Preencher valores categóricos com a moda
    colunas_categoricas = df.select_dtypes(include=['object']).columns
    for col in colunas_categoricas:
        df[col] = df[col].fillna(df[col].mode()[0])  # Preenchendo com moda diretamente

def verifica_colunas_numericas(df):
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    todas_numericas = len(colunas_numericas) == df.shape[1]

    if not todas_numericas:
        colunas_nao_numericas = df.columns[~df.columns.isin(colunas_numericas)]
        print("As seguintes colunas não são numéricas:", colunas_nao_numericas.tolist())

    return todas_numericas

def KNN(treino, teste):
    # Preenchendo valores faltantes
    bota_faltantes(treino)
    bota_faltantes(teste)

    # Normalizando dados
    treino = normaliza_palavra(treino)
    teste = normaliza_palavra(teste)

    # Separando características e rótulos
    X_train = treino.drop(columns=['Survived'])  # Supondo que 'Survived' é a coluna alvo
    y_train = treino['Survived']

    # Normalizando números
    X_train = normaliza_numeros(X_train)
    X_test = normaliza_numeros(teste)

    # Remover colunas não numéricas
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # Verificação se todas as colunas de X_train e X_test são numéricas
    if not verifica_colunas_numericas(X_train) or not verifica_colunas_numericas(X_test):
        raise ValueError("Algumas colunas não são numéricas após a normalização.")

    # Garantir que não haja valores NaN
    if X_train.isnull().values.any() or X_test.isnull().values.any():
        raise ValueError("Valores NaN encontrados em X_train ou X_test.")

    # Verificação se as colunas estão alinhadas
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("O número de colunas em X_train e X_test não corresponde.")

    # Criação do modelo
    model = KNeighborsClassifier(n_neighbors=2)

    # Treinamento do modelo
    model.fit(X_train, y_train)

    # Previsão
    predictions = model.predict(X_test)

    return predictions

def analise(predictions, respostas):

    if 'Survived' not in respostas.columns:
        raise ValueError("O DataFrame de resposta não contém a coluna 'Survived'.")

    acerto = np.sum(predictions == respostas['Survived'].values)
    print(f"A taxa de acerto foi: {acerto / len(predictions):.2f}")

# Executando a previsão e análise
previsao = KNN(treino, teste)
analise(previsao, respostas)

gerar_csv_previsao(teste, previsao)
