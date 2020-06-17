import numpy as np
from BatchModel import Data
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        u = y.iloc[i:(i + time_steps)].values
        ys.append(u)

    return np.array(Xs), np.array(ys)


TIME_STEPS = 30
TRAIN_SIZE = .80
THRESHOLD = 1.5

data = Data()
data.fetch_data(path="data\HK_anomalies_14_4_and_3_5\HK")
df = data.raw_dataset
train_size = int(len(df) * TRAIN_SIZE)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

print(train.shape, test.shape)

scaler = StandardScaler()
scaler = scaler.fit(train)
train = pd.DataFrame(scaler.transform(train))
test = pd.DataFrame(scaler.transform(test))

print(train.shape, test.shape)

X_train, y_train = create_dataset(train, train, TIME_STEPS)
X_test, y_test = create_dataset(test, test, TIME_STEPS)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# LSTM Autoencoder

model = keras.Sequential()
model.add(keras.layers.LSTM(units=128,
                            input_shape=(X_train.shape[1], X_train.shape[2])
                            ))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.LSTM(
    units=128,
    return_sequences=True
))
model.add(keras.layers.Dropout(rate=0.2))

model.add(keras.layers.TimeDistributed(keras.layers.Dense(
    units=X_train.shape[2]
)))
model.add(keras.layers.Activation('relu'))

model.compile(loss='mse', optimizer='adam')

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred, X_train), axis=1)
sns.distplot(train_mae_loss, bins=50)
plt.show()

X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred, X_test), axis=1)

plt.subplots(figsize=(11, 11))
fet1, = plt.plot(X_test[0:2114, 0, 0])
fet2, = plt.plot(X_test[0:2114, 0, 1])
fet3, = plt.plot(X_test[0:2114, 0, 2])
fet4, = plt.plot(X_test[0:2114, 0, 3])
fet5, = plt.plot(X_test[0:2114, 0, 4])
fet6, = plt.plot(X_test[0:2114, 0, 5])
plt.legend([fet1, fet2, fet3, fet4, fet5, fet6, ], data.feacher_names)
plt.title("Real data")
plt.show(block=False)

plt.subplots(figsize=(11, 11))
fet1, = plt.plot(X_test_pred[0:2114, 0, 0])
fet2, = plt.plot(X_test_pred[0:2114, 0, 1])
fet3, = plt.plot(X_test_pred[0:2114, 0, 2])
fet4, = plt.plot(X_test_pred[0:2114, 0, 3])
fet5, = plt.plot(X_test_pred[0:2114, 0, 4])
fet6, = plt.plot(X_test_pred[0:2114, 0, 5])
plt.legend([fet1, fet2, fet3, fet4, fet5, fet6, ], data.feacher_names)
plt.title("Predicted data")
plt.show(block=False)

