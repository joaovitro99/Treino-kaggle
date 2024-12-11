import pandas as pd

# Um DataFrame é uma planilha/matriz - abaixo usamos dicionário para criá-lo
pd.DataFrame({'yes': [50, 21], 'no': [12, 9]})

# Nomeamos colunas e também linhas
pd.DataFrame({'yes': [50, 21], 'no': [12, 9]}, index=['happy', 'sad'])

# Uma Série é basicamente um vetor
pd.Series([1, 1, 2, 3])

# Indexar elementos
pd.Series([1, 1, 2, 3], index=['filhos de bob', 'filhos de bobb', 'filhos de bbb', 'filhos de obb'])

# Ler CSV
titanic = pd.read_csv('nam.csv')

# Retorna número de linhas e número de colunas do arquivo
titanic.shape

# Pegar as primeiras linhas da tabela (cabeça)
titanic.head()

# Pegar as últimas linhas da tabela
titanic.tail()

# Faz a coluna 0 ser o índice
titanic = pd.read_csv('nam.csv', index_col=0)

# **Novas funções úteis**

# Exibir informações gerais do DataFrame, como tipos de dados e valores nulos
titanic.info()

# Estatísticas descritivas de colunas numéricas
titanic.describe()

# Selecionar colunas específicas
titanic['Survived']  # Seleciona uma única coluna
titanic[['Survived', 'Age']]  # Seleciona múltiplas colunas

# Filtrar linhas com base em uma condição
titanic[titanic['Age'] > 30]  # Passageiros com mais de 30 anos

# Criar uma nova coluna
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1

# Ordenar os dados por uma coluna
titanic.sort_values(by='Age', ascending=False)

# Remover colunas
titanic.drop(columns=['Cabin'], inplace=True)

# Lidando com valores nulos
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)  # Substitui valores nulos pela média
titanic.dropna(inplace=True)  # Remove linhas com valores nulos

# Aplicar uma função em uma coluna
titanic['AgeGroup'] = titanic['Age'].apply(lambda x: 'Child' if x < 18 else 'Adult')

# Agrupar dados e calcular estatísticas
titanic.groupby('Survived')['Age'].mean()  # Média de idade por sobrevivência

# Salvar o DataFrame como CSV
titanic.to_csv('titanic_cleaned.csv', index=False)


# **Função `map`**
# Aplica uma função a cada elemento de uma Série.

# Exemplo: converter valores de uma coluna em outra representação
df = pd.DataFrame({'Names': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]})
# Criar uma nova coluna que classifica a idade
df['AgeGroup'] = df['Age'].map(lambda x: 'Young' if x < 30 else 'Adult')
print(df)
# Resultado:
#      Names  Age AgeGroup
# 0    Alice   25    Young
# 1      Bob   30    Adult
# 2  Charlie   35    Adult


# **Função `groupby`**
# Agrupa os dados com base em valores de uma coluna e calcula estatísticas.

# Exemplo: calcular a média de idade por grupo
df = pd.DataFrame({'Department': ['HR', 'HR', 'IT', 'IT'], 'Age': [25, 30, 22, 32]})
grouped = df.groupby('Department')['Age'].mean()
print(grouped)
# Resultado:
# Department
# HR    27.5
# IT    27.0
# Name: Age, dtype: float64


# **Função `join`**
# Combina dois DataFrames com base no índice ou em uma chave específica.

# Exemplo: juntar dois DataFrames usando os índices
df1 = pd.DataFrame({'Salary': [5000, 7000]}, index=['Alice', 'Bob'])
df2 = pd.DataFrame({'Age': [25, 30]}, index=['Alice', 'Bob'])
joined = df1.join(df2)  # Combina os DataFrames com base no índice
print(joined)
# Resultado:
#        Salary  Age
# Alice    5000   25
# Bob      7000   30

# Exemplo: combinar DataFrames com uma chave
df1 = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Salary': [5000, 7000]})
df2 = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})
joined = df1.merge(df2, on='Name')  # Combina com base na coluna "Name"
print(joined)
# Resultado:
#    Name  Salary  Age
# 0  Alice    5000   25
# 1    Bob    7000   30
