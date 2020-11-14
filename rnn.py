import pandas as pd
import numpy as np

# Raw url for stock data 
url = "https://raw.githubusercontent.com/AbedMir31/StockMarketRNN/main/SPY%20(1).csv"

# Preprocess the dataset 
headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data = pd.read_csv(url, sep = ',', names = headers, skiprows=1)
data = data.drop('Adj Close', axis=1)
data = data.drop('Date', axis=1)

print(data)

X = [data['Open'], data['High'], data['Volume']]
Y = [data['Close']]
X = np.array(X)
Y = np.array(Y)

# Holds all relevant attributes for the prediction
X_train = np.array(X)
# Trying to predict the closing price for the next day
y_train = np.array(Y)


learning_rate = .0001
epoch = 25
T = len(data)
hidden_dim = 100
output_dim = 1

backprop_trunc = 5
min_clip = -10
max_clip = 10

# Weights
U = np.random.uniform(0, 1, (hidden_dim, T))
W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
V = np.random.uniform(0, 1, (output_dim, hidden_dim))

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for e in range(epoch):
    loss = 0.0

    #forward pass
    for i in range(Y.shape[0]):
        x = X[i]
        y = Y[i]
        prev_state = np.zeros((hidden_dim, 1))
        for t in range(T):
            newinput = np.zeros(x.shape)
            newinput[t] = x[t]
            mulu = np.dot(U, newinput)
            mulw = np.dot(W, prev_state)
            add = mulw + mulu
            s = sigmoid(add)
            mulv = np.dot(V, s)
            prev_state = s
    
    # calculate error 
        loss_per_record = (y - mulv)**2 / 2
        loss += loss_per_record
    loss = loss / float(y.shape[0])
    print(loss)
