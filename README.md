# LGMVIP-DATA-SCIENCE-TASK-05

CODE:

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:/Users/WELCOME/OneDrive/Documents/INTERNSHIP/NSE-TATAGLOBAL.csv")
data

plt.plot(data["Date"],data['High'])
plt.xlabel("Date")
plt.ylabel("High")

sns.lineplot(x=data["Date"],y=data['Turnover (Lacs)'])

scale = MinMaxScaler(feature_range=(0,1))
scaled_data = scale.fit_transform(data['Close'].values.reshape(-1,1))

train_size = int(len(scaled_data) * 0.75)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_seq(data,seq_length):
    X = []
    Y = []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        Y.append(data[i+seq_length])
    return np.array(X),np.array(Y)

seq_length = 20
X_train,Y_train = create_seq(train_data,seq_length)
X_test,Y_test = create_seq(test_data,seq_length)

model = Sequential()
model.add(LSTM(50,return_sequences = True, input_shape = (seq_length,1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer = 'adam',loss='mean_squared_error')

model.fit(X_train,Y_train,epochs=50,batch_size=40)

train_loss = model.evaluate(X_train,Y_train,verbose = 0)
test_loss = model.evaluate(X_test,Y_test,verbose = 0)
print(f"Train Loss: {train_loss: .4f}")
print(f"Test Loss: {test_loss: .4f}")

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_predictions = scale.inverse_transform(train_predictions)
test_predictions = scale.inverse_transform(test_predictions)

print("Predicted Values for the Training Set: ")
print(train_predictions.flatten())

print("Predicted Values for the Testing Set: ")
print(test_predictions.flatten())
