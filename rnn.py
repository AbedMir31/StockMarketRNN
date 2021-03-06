#John Kenney (jfk150030),  Matthew Brown (meb180001), Abed Mir (amm161930), Saman Laleh (sxl173130)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt 
import plotly.graph_objects as go
import plotly.graph_objects as go
import datetime as dt
import matplotlib.dates as mdates
logger = open("log.txt", "a+")

class RNN:
    def __init__(self,url,Timesteps,Learning_Rate,Max_Iterations,Hidden_Layers):
        self.URL = url
        self.timestep = Timesteps
        self.learning_rate = Learning_Rate
        self.max_iter = Max_Iterations
        self.hidden_layer = Hidden_Layers

        headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        data = pd.read_csv(url, sep = ',', names = headers,skiprows = 1)
        data = data.drop('Adj Close', axis=1)

        training_size = .8
        train_dataset, test_dataset = train_test_split(data,train_size = training_size, shuffle = False)
        scaler = MinMaxScaler()
        scaler2 = MinMaxScaler()

        TestDays = test_dataset.iloc[self.timestep:,0]
        TestDays = TestDays.reset_index().drop('index', axis=1)
        TestDays = TestDays.iloc[:,0].str.replace('-','/')
        self.date = [dt.datetime.strptime(d,'%Y/%m/%d').date() for d in TestDays]

        train_dataset = train_dataset.drop('Date', axis=1)
        test_dataset = test_dataset.drop('Date', axis=1)
        train_dataset = scaler.fit_transform(train_dataset.values)
        test_dataset = scaler2.fit_transform(test_dataset.values)
        # print(train_dataset)
        
        self.X = []
        self.Y = []
        self.Xtest = []
        self.Ytest = []
        self.input_layer = 1
        self.data_size = len(data)

        for i in range(self.timestep,len(train_dataset)):
            self.X.append(train_dataset[i - self.timestep:i,3])#[0,1,2,4]])
            self.Y.append(train_dataset[i,3])
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

        for i in range(self.timestep,len(test_dataset)):
            self.Xtest.append(test_dataset[i - self.timestep:i,3])#[0,1,2,4]])
            self.Ytest.append(test_dataset[i,3])
        self.Xtest = np.array(self.Xtest)
        self.Ytest = np.array(self.Ytest)

        # Weights
        self.U = np.random.uniform(0,1,(self.input_layer, self.hidden_layer))
        self.V = np.random.uniform(0,1,(self.hidden_layer, self.input_layer))
        self.W = np.random.uniform(0,1,(self.hidden_layer, self.hidden_layer))

        # Bias layers, bias_hidden and bias_output
        self.bh = np.random.uniform(0,1,(self.input_layer, self.hidden_layer))
        self.bo = np.random.uniform(0,1,(self.input_layer, self.input_layer))

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh_derivative(self, x):
        return (1 - np.power(x,2))

    def forward_propagate(self, inputs, hidden_state):
        out = 0
        for t in range(len(inputs)):
            # CODE - compute new hidden state use old hidden state
            # h  = tanh(U*x + h(t-1)*W + bh)
            hidden_state[t+1] = self.tanh(np.dot(self.U,inputs[t]) + np.dot(hidden_state[t],self.W) + self.bh)
            # CODE - compute output
            # out = sigmoid(h(t-1)V + bo)
            out = self.sigmoid(np.dot(hidden_state[t+1],self.V) + self.bo)
        return out, hidden_state

    def bptt(self, example,y,pred,h,t):
        dL = self.derv_loss(y,pred)
        dV = dL* self.sigmoid_derivative(pred)*h[t].T

        dW = dL* self.sigmoid_derivative(pred)*np.multiply(self.V,self.derv_hW(example,h,t))

        dU = dL* self.sigmoid_derivative(pred)*np.dot(self.V.T,self.derv_hU(example,h,t))
        
        dbo = dL*self.sigmoid_derivative(pred)

        dbh = dL* self.sigmoid_derivative(pred)*np.dot(self.V.T,self.derv_bh(example,h,t))

        return dV, dW, dU, dbo, dbh

    def derv_loss(self, y,pred):
        return -2*(y - pred)

    def derv_hW(self, example,h,t):
        if t == 1:
            return np.zeros((self.input_layer,self.hidden_layer))
        else:
            return np.multiply(self.tanh_derivative(h[t]),(h[t]+ np.multiply(self.W,self.derv_hW(example,h,t-1))))

    def derv_hU(self, example,h,t):
        if t == 1:
            return np.multiply(self.tanh_derivative(h[t]),example[0])
        else:
            return np.multiply(self.tanh_derivative(h[t]),(example[t-1]+ np.multiply(self.W,self.derv_hU(example,h,t-1))))

    def derv_bh(self, example,h,t):
        if t == 1:
            return np.multiply(self.tanh_derivative(h[t]),1)
        else:
            return np.multiply(self.tanh_derivative(h[t]),(1+ np.multiply(self.W,self.derv_bh(example,h,t-1))))


    def update_weights(self, dV,dW,dU,dbo,dbh):
        self.V -= self.learning_rate*dV
        self.W -= self.learning_rate*dW
        self.U -= self.learning_rate*dU
        self.bo -= self.learning_rate*dbo
        self.bh -= self.learning_rate*dbh

    def loss(self, out, y):
        # Least squares (MSE)
        return np.power((out - y),2)

    def train(self):
        for i in range(0,self.max_iter):
            hidden_state = []
            for example in range(len(self.X)):
                for i in range(0,self.timestep+1):
                    hidden_state.append(np.zeros((self.input_layer,self.hidden_layer)))
                o, h = self.forward_propagate(self.X[example], hidden_state)
                L = self.loss(o,self.Y[example])
                '''
                print('Loss', L)
                print("Predicted: ", o)
                print("true Y: ", self.Y[example])
                '''
                dV,dW,dU,dbo,dbh = self.bptt(self.X[example],self.Y[example],o,hidden_state,self.timestep)
                self.update_weights(dV,dW,dU,dbo,dbh)
            
    def test(self):
        hidden_state = []
        time = np.arange(len(self.Ytest))
        pred = np.zeros(len(self.Xtest))
        L = np.zeros(len(self.Xtest))
        for example in range(len(self.Xtest)):
            for i in range(0,self.timestep+1):
                hidden_state.append(np.zeros((self.input_layer, self.hidden_layer)))
            pred[example], h = self.forward_propagate(self.Xtest[example], hidden_state)
            L[example] = self.loss(pred[example],self.Ytest[example])  
        MSE = mean_squared_error(self.Ytest,pred)
        num = np.arange(len(pred))
        logger.write("RNN Timestep = {}, learning rate = {}, iterations = {}, # hidden layers = {}, Error Function = RMSE \t\t\t Train/Test Split: 80/20, Size of dataset = {}, MSE = {}, RMSE = {}".format(self.timestep,self.learning_rate,self.max_iter,self.hidden_layer, self.data_size, round(MSE,7), np.sqrt(MSE)))
        logger.write("\n")
        self.print_results(num, pred, L,MSE)
    def print_results(self, num, pred, L, MSE):
        plt.title("Stock Prices Vs Date") 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=50))
        plt.xlabel("Date") 
        plt.ylabel("Stock prices") 
        plt.plot(self.date,self.Ytest,"b",label = "true")
        plt.plot(self.date,pred,"r",label = "predicted") 
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.gcf().autofmt_xdate()
        plt.show()    

        plt.title("Error Vs Date") 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=50))
        plt.xlabel("Date") 
        plt.ylabel("Error") 
        plt.plot(self.date,L,"b",label = "Error")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.gcf().autofmt_xdate()
        plt.show()

        fig = go.Figure(data=[go.Table(header=dict(values=['timestep', 'Learning rate','max iterations','hidden layers','MSE']),
                    cells=dict(values=[self.timestep,self.learning_rate,self.max_iter,self.hidden_layer,round(MSE,7)]))
                        ])
        fig.show()  

# Set parameters to whichever values you prefer
Url = "https://raw.githubusercontent.com/AbedMir31/StockMarketRNN/main/SPY.csv"
Timesteps = 2
Learning_Rate = .001
Max_Iterations = 100
Hidden_Layers = 5
rnn = RNN(Url,Timesteps,Learning_Rate,Max_Iterations,Hidden_Layers)
rnn.train()
rnn.test()