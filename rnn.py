import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Raw url for stock data 
url = "https://raw.githubusercontent.com/AbedMir31/StockMarketRNN/main/SPY.csv"

training_size = .8
train_data = pd.read_csv(url)
# training_data, test_data = train_test_split(data, train_size = training_size, shuffle = False)
#print(train_data.head)
train_data = train_data.iloc[:,4].values
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data.reshape(-1,1))

timestep = 40
x_train = []
y_train = []
for i in range (timestep, len(train_data)):
    x_train.append(train_data[i-timestep:i, 0])
    y_train.append(train_data[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

price_dim = len(train_data)
hidden_dim = 100
bptt_truncate = 4

U = np.random.uniform(-np.sqrt(1./price_dim), np.sqrt(1./price_dim), (hidden_dim, price_dim))
V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (price_dim, hidden_dim))
W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

# We want to use the softmax activation function in the RNN
def softmax(x):
    e = np.exp(x)
    return e / e.sum()

def forward_propagate(x):
    T = timestep
    s = np.zeros((T+1, hidden_dim))
    s[-1] = np.zeros(hidden_dim)
    o = np.zeros((T, price_dim))
    for i in np.arange(T):
        s[i] = np.tanh(U[:,int(x[i])] + W.dot(s[i-1]))
        o[i] = softmax(V.dot(s[i]))
    return [o, s]

def predict_state(x):
    o, s = forward_propagate(x)
    return np.argmax(o, axis=1)

predict = predict_state(x_train[10])
print(predict.shape)
print(predict)