import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('original.csv')

L = len(df)
print(L)

Y = np.array([df.loc[:,'Close']])

plt.figure(1)
plt.plot(Y[0,:])
plt.show(block=False)


X1 = Y[:,0:L-5]
X2 = Y[:,1:L-4]
X3 = Y[:,2:L-3]

X = np.concatenate([X1,X2,X3],axis=0)
X = np.transpose(X)
print(X)

Y=np.transpose(Y[:,3:L-2])
#normalize [-1,1]
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

scaler1 = MinMaxScaler()
scaler1.fit(Y)
Y = scaler1.transform(Y)

X = np.reshape(X, (X.shape[0],1,X.shape[1]))

l = train_test_split(X,Y,test_size=0.25)
X_train = l[0]
X_test = l[1]
Y_train = l[2]
Y_test = l[3]

model = Sequential()
model.add(LSTM(10,activation='tanh',input_shape=(1,3),recurrent_activation='hard_sigmoid'))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'rmsprop',metrics = [metrics.mae])

model.fit(X_train, Y_train, epochs=100, verbose=2)

Predict = model.predict(X_test)

plt.figure(2)
plt.scatter(Y_test,Predict)
plt.show(block = False)

plt.figure(3)
Test, = plt.plot(Y_test)
Predict, = plt.plot(Predict)
plt.legend([Predict,Test],["Predicted Data", "Real Data"])
plt.show()