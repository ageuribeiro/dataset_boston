# Importando as bibliotecas básicas

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Carregando a Base de Dados

# carrega o dataset de boston
from sklearn.datasets import load_boston
boston = load_boston()


# descrição do dataset
# print(boston.DESCR)

# cria um dataframe pandas
data = pd.DataFrame(boston.data, columns=boston.feature_names)

# imprime as 5 primeiras linhas do dataset
# print(data.head(15))

# escreve o arquivo para o disco
data.to_csv('data.csv')


# Adicionando a coluna que será nossa variável alvo.
# adiciona a variável MEDV

data["MEDV"] = boston.target

# imprime as primeiras 15 linhas do dataframe
# print(data.head(15))


# mostra a descrição do dataset
# print(data.describe())

# ANÁLISE E EXPLOCARAÇÃO DO DADOS

# instale o pandas profiling pip install pandas-profiling

# import o ProfileReport
from pandas_profiling import ProfileReport

# executando o profile
profile = ProfileReport(data, title='Relatório - Pandas Profiling', html={'style':{'full_width':True}})

# salvando o relatório no disco
profile.to_file(output_file="Relatorio01.html")

#Check missing values
data.isnull().sum()

#calculando a correlação
correlacao = data.corr()

#usando o método heatmap do seaborn
# %matplotlib inline
plt.figure(figsize=(16,6))
sns.heatmap(data = correlacao, annot=True)

# importando o PlotLy
import plotly.express as px

# RM vs MEDV (Número de quartos e valor médio do ímovel)
# fig = px.scatter(data, x=data.RM, y=data.MEDV)
# fig.show()

# LSTAT vs MEDV (índice de status mais baixo da população e preço de imóvel)
# fig = px.scatter(data, x=data.LSTAT, y=data.MEDV)
# fig.show()

# PTRATIO vs MEDV (percentual de proporção de alunos para professores e o valor médio de imóveis)
# fig = px.scatter(data, x=data.PTRATIO, y=data.MEDV)
# fig.show()

# ANALISANDO OUTLIERS

# estatśtica descritiva da variável RM
data.RM.describe()

#visualizando a distribuição da variável RM
import plotly.figure_factory as ff

labels = ['Distribuição da variável RM (números de quartos)']
#fig = ff.create_distplot([data.RM], labels, bin_size=.2)
#fig.show()

#Visualizando outliers na variavel RM
import plotly.express as px
fig = px.box(data, y='RM')
#fig.update_layout(width=800, height=800)
#fig.show()

#ANALISANDO A SIMETRIA DO DADO

# carrega o método stats da scipy
from scipy import stats

# imprime o coeficiente de pearson
print(stats.skew(data.MEDV))

# Histograma da variável MEDV (variável alvo)
fig = px.histogram(data, x="MEDV", nbins=50, opacity=0.50)
fig.show()




