import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics

df = pd.read_csv('original.csv')

TAVG = np.array([df.iloc[:,6]])
TMAX = np.array([df.iloc[:,5]])
TMIN = np.array([df.iloc[:,4]])

#print(TAVG)

fig = plt.figure(1)
ax = fig.add_subplot(111,projection='3d')

ax.scatter(TMAX,TMIN,TAVG,marker='o')
ax.set_xlabel('TMIN')
ax.set_ylabel('TMAX')
ax.set_zlabel('TAVG')
plt.show(block=False)

plt.figure(2)
plt.plot(TAVG[0,:])
plt.show(block=False)

X = np.concatenate([TMIN,TMAX],axis=0)
X = np.transpose(X)

Y = np.transpose(TAVG)


sc = MinMaxScaler()
sc.fit(X)
X = sc.transform(X)

sc1 = MinMaxScaler()
sc1.fit(Y)
Y = sc1.transform(Y)

X = np.reshape(X,(X.shape[0],1,X.shape[1]))


l = train_test_split(X,Y,test_size=0.3)
X_train = l[0]
X_test = l[1]
Y_train = l[2]
Y_test = l[3]

model = Sequential()

model.add(LSTM(20,activation='tanh',input_shape=(1,2),recurrent_activation='hard_sigmoid'))

model.add(Dense(1))

model.compile(loss='mse',optimizer='rmsprop',metrics=[metrics.mae])

model.fit(X_train,Y_train,epochs=50,verbose=2)

predict = model.predict(X_test)

plt.figure(3)
plt.scatter(Y_test,predict)
plt.show(block = False)

plt.figure(4)
Real, = plt.plot(Y_test)
Predict, = plt.plot(predict)
plt.legend([Real,Predict],["Real Data","Predicted Data"])
plt.show()
