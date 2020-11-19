import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Raw url for stock data 
url = "https://raw.githubusercontent.com/AbedMir31/StockMarketRNN/main/SPY.csv"

headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data = pd.read_csv(url, sep = ',', names = headers,skiprows = 1)
data = data.drop('Adj Close', axis=1)
data = data.drop('Date', axis=1)

training_size = .8
train_dataset, test_dataset = train_test_split(data,train_size = training_size, shuffle = False)
scaler = MinMaxScaler()
train_dataset = scaler.fit_transform(train_dataset.values)
# print(train_dataset)

X = []
Y = []
timestep = 20
for i in range(timestep,len(train_dataset)):
  X.append(train_dataset[i - timestep:i,3])#[0,1,2,4]])
  Y.append(train_dataset[i,3])
X = np.array(X)
Y = np.array(Y)
'''
print('X', X.shape)
print('Y', Y.shape)
print('X',X[0])
'''

input_layer = 1
hidden_layer = 100

# Weights
U = np.random.randn(input_layer, hidden_layer)
V = np.random.randn(hidden_layer, input_layer)
W = np.random.randn(hidden_layer, hidden_layer)

# Bias layers, bias_hidden and bias_output
bh = np.random.randn(input_layer, hidden_layer)
bo = np.random.randn(input_layer, input_layer)


# We want to use the softmax activation function in the RNN
def softmax(x):
    e = np.exp(x)
    return e / e.sum()
def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def forward_propagate(inputs, hidden_state):
    outputs, hidden_states = [], []
    for t in range(len(inputs)):
        # CODE - compute new hidden state use old hidden state
        # temp  = U*x + h(t-1)*W +bh

        temp1 = np.dot(U,inputs[t])
        temp2 = np.dot(hidden_state[t],W) + bh
        hidden_state[t+1] = tanh(temp1 + temp2)
        # CODE - compute output
        
        out = np.dot(hidden_state[t+1],V) + bo

        outputs.append(out)
        hidden_states.append(hidden_state[t+1].copy())
    return outputs, hidden_state
'''
def back_propagate(x):
    
def update_weights(x):
'''
def train(x):
    hidden_state = []
    for example in range(0,2):
        for i in range(0,timestep+1):
            hidden_state.append(np.zeros((input_layer,hidden_layer)))
        o, h = forward_propagate(x[example], hidden_state)
        '''
        back_propagate()
        update_weights()
        '''
train(X)