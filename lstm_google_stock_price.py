# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 08:46:20 2020

DESAFIO: PREDIZER O PREÇO DAS AÇÕES DO GOOGLE

Método: LSTM
In: Preço das ações de 2004 até hoje
Out: Tendência do preço da ação para os próximos 20 dias

@author: celin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########## Importing data
dataset = pd.read_csv('GOOG.csv')
# Separando em dados de treinamento e de teste
train_dataset = dataset.iloc[0:(len(dataset)-20),:]
test_dataset = dataset.iloc[(len(dataset)-20):,:]

# Visualizing stock opening data
xTrain_data = train_dataset.iloc[:,0].values
yTrain_data = train_dataset.iloc[:,1:2].values

xTest_data = test_dataset.iloc[:,0].values;
yTest_data = test_dataset.iloc[:,1:2].values;

# plt.figure(1)
# plt.plot(xTrain_data,yTrain_data);
# plt.figure(2)
# plt.plot(xTest_data,yTest_data)
# plt.show();

########## Feature Scaling: 
# RNN com sigmoid activation function -> Normalização
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1),copy=True)
scaled_trained_data = sc.fit_transform(yTrain_data)

##### Creating a data structure with 60 timestamp and 1 output
# Isso significa que a cada 60 passos a rede vai tentar entender o padrão e 
# gerar uma saída. 60 foi um número aleatório, escolhido com base em testes.
# Ou seja, a rede tenta entender o padrão de 60 em 60 dias (3 meses) e predizer 
# o valor do dia seguinte.

x_train = []
y_train = []

# Cria uma matriz com os valores para cada dia (60 por dia)
for i in range(60,len(yTrain_data)):
    x_train.append(scaled_trained_data[i-60:i,0])
    y_train.append(scaled_trained_data[i,0]) # preço em t+1

x_train, y_train = np.array(x_train), np.array(y_train)

##### Reshaping
# Adiciona uma nova dimensão no array
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))
# 1 dim: numero de preços das ações
# 2 dim: numero de time steps
# 3 dim: Numero de indicadores (variáveis que influeciam)

########## Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

##### Initialising the RNN
# Regressão é sobre predizer um valor contínuo
regressor = Sequential();

# Adding the first LSTM layer and droupout regularisation
regressor.add(LSTM(
    units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)
    ))
regressor.add(Dropout(0.2)) # dropout=20%, ignora 20% dos neuronios durante cada iteração do treinamento

# Adding the second LSTM layer and droupout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) 

# Adding the third LSTM layer and droupout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) 

# Adding the fourth LSTM layer and droupout regularisation
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2)) 

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(x_train, y_train, batch_size = 32, epochs = 120)

########## Making the prediction and visualising the results

##### Getting the predicted stock price of january 2017
# Pra predizer o valor dos próximos 30 dias precisa dos valores dos últimos 6 meses
# pra isso eu preciso tanto dos dados de teste como dos de treinamento, pois podem
# haver valores nessas datas presentes nos dois datasets.
# Logo eu preciso concatenar os dados de teste e de treinamento
dataset_total = pd.concat((train_dataset['Open'],test_dataset['Open']),
                          axis = 0) #axis = 0 -> concatena na vertical
inputs = dataset_total[len(dataset_total) - len(test_dataset) - 60 : ].values
inputs = inputs.reshape(-1,1) # inline e 1 coluna

# Normalizar: Não vai usar o fit_transform pois é preciso usar a mesma transformação
# usada anteriormente. Portanto, só vai usar transform.
inputs = sc.transform(inputs) 

# Cria uma matriz com os valores para cada dia (60 por dia)
x_test = []
# 60, 3 meses, + 20, que são os dias válidos de um mes. logo, 80
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])

x_test = np.array(x_test)

# Adiciona uma nova dimensão no array
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(yTest_data, color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue',label='Predicted Google Stock Price')
plt.title('Google stock price prediction - epochs = 120')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()


