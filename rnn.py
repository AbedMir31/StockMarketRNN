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

input_layer = 1
hidden_layer = 100

'''
U = [input layers,hidden layers]
W = [hidden layers,hidden layers]
V = [hidden layers,input layers]
'''

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh_derivative(x):
    return (1 - np.power(x,2))

def forward_propagate(inputs, hidden_state):
    for t in range(len(inputs)):
        # CODE - compute new hidden state use old hidden state
        # temp  = U*x + h(t-1)*W +bh

        temp1 = np.dot(U,inputs[t])
        temp2 = np.dot(hidden_state[t],W) + bh
        hidden_state[t+1] = tanh(temp1 + temp2)
        # CODE - compute output
        
        out = sigmoid(np.dot(hidden_state[t+1],V) + bo)

    return out, hidden_state

def back_propagate(inputs, outputs, hidden_states, targets):

    #Initialize gradients to zero
    d_U, d_V, d_W = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
    d_b_h, d_b_o = np.zeros_like(bh), np.zeros_like(bo)

    loss = 0
    d_h_next = np.zeros_like(hidden_states[0])
    for t in reversed(len(outputs)):
        epsilon = .0001

        # Compute cross-entropy loss (as a scalar)
        # When taking logarithms, it's a good idea to add a small constant (e.g. 1e-9)
        loss += loss(outputs[t], targets[t])

        # Backpropagate into output (derivative of cross-entropy)
        # If you're confused about this step, see this link for an explanation:
        d_o = outputs[t].copy()
        d_o[np.argmax(targets[t])] -= 1
        
        # Backpropagate into W
        # YOUR CODE HERE!
        d_W += 
        d_b_out += d_o
        
        # Backpropagate into h
        d_h = np.dot(W.T, d_o) + d_h_next
        
        # Backpropagate through non-linearity
        # (we assume tanh is used here)
        d_f = (1 - hidden_states[t]**2) * d_h
        d_b_hidden += d_f
        
        # Backpropagate into U
        # YOUR CODE HERE!
        d_U += 
        
        # Backpropagate into V
        # YOUR CODE HERE!
        d_V += 
        d_h_next = np.dot(V.T, d_f)
    
    # Pack gradients
    grads = d_U, d_V, d_W, d_b_h, d_b_o    
    
    # Clip gradients
    grads = clip_gradient_norm(grads)
    
    return loss, grads

'''
def update_weights(x):
'''
def loss(out, y):
    # Least squares (MSE)
    return np.power((out - y),2)

def train(x):
    hidden_state = []
    for example in range(0,1):
        for i in range(0,timestep+1):
            hidden_state.append(np.zeros((input_layer,hidden_layer)))
        o, h = forward_propagate(x[example], hidden_state)
        L = loss(o,Y[example])
        print('Loss', L)
        print("OUTPUT: ", o)
        '''
        back_propagate()
        update_weights()
        '''
train(X)