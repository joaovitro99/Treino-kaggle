import numpy as np

# **Criação de Arrays para Análise de Dados**
data = np.array([[1.5, 2.3, 3.1], [4.0, 5.2, 6.8]])  # Matriz de dados
zeros = np.zeros((3, 4))  # Matriz 3x4 inicializada com zeros
ones = np.ones((3, 4))  # Matriz 3x4 inicializada com uns

# **Estatísticas básicas**
mean = data.mean(axis=0)  # Média por coluna
std = data.std(axis=1)  # Desvio padrão por linha
sum_total = data.sum()  # Soma total dos elementos
min_val = data.min()  # Valor mínimo
max_val = data.max()  # Valor máximo

# **Manipulação de Dados**
reshape = data.reshape((3, 2))  # Reorganizar matriz para 3x2
transpose = data.T  # Transpor os dados (linhas viram colunas)
flatten = data.flatten()  # Transformar em vetor 1D

# **Indexação e Filtragem**
row_1 = data[0]  # Seleciona a primeira linha
value_filter = data[data > 3]  # Filtra valores maiores que 3
col_2 = data[:, 1]  # Seleciona a segunda coluna

# **Operações em Dados**
scaled = data * 10  # Multiplica todos os valores por 10
log_data = np.log(data + 1)  # Logaritmo natural (evita log de 0 adicionando 1)
dot_prod = np.dot(data, [[1], [0], [1]])  # Produto escalar

# **Trabalho com Dados Aleatórios**
random_data = np.random.rand(100, 5)  # Dataset aleatório com 100 linhas e 5 colunas
random_ints = np.random.randint(0, 100, (10, 3))  # Inteiros aleatórios entre 0-99

# **Identificar Dados Faltantes**
nan_array = np.array([1, 2, np.nan, 4])
nan_mask = np.isnan(nan_array)  # Retorna True para valores NaN
nan_sum = np.nansum(nan_array)  # Soma ignorando NaNs
