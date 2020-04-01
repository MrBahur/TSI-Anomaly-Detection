import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import datetime
import glob


class Reader:
    def __init__(self, path):
        self.path = path

    def set_path(self, path):
        self.path = path

    # read all the csv file in self.path
    # return Data Frame containing all of the data
    def read(self):
        all_filenames = glob.glob(self.path + "/*.csv")
        data_types = {'ds': str, 'y': float}
        parse_dates = ['ds']
        frame = pd.concat(
            (
                pd.read_csv(filename, dtype=data_types, parse_dates=parse_dates, date_parser=pd.to_datetime,
                            index_col=None,
                            header=0) for filename in all_filenames),
            axis=0, ignore_index=True)
        return frame


class MyModel:

    def __init__(self):
        self.DATA = ('total_success_action_conversions',
                     'recommendation_requests_5m_rate_dc',
                     'total_failed_action_conversions',
                     'trc_requests_timer_p95_weighted_dc',
                     'trc_requests_timer_p99_weighted_dc')

    # fetching the data from path,
    # where path is the main folder containing the data.
    def fetch_data(self, path):
        reader = Reader(path=path + '//' + self.DATA[0])
        self.target = np.array([reader.read().loc[:, 'y']])
        reader.set_path(path=path + '//' + self.DATA[1])
        self.feature1 = np.array([reader.read().loc[:, 'y']])
        reader.set_path(path=path + '//' + self.DATA[2])
        self.feature2 = np.array([reader.read().loc[:, 'y']])
        reader.set_path(path=path + '//' + self.DATA[3])
        self.feature3 = np.array([reader.read().loc[:, 'y']])
        reader.set_path(path=path + '//' + self.DATA[4])
        self.feature4 = np.array([reader.read().loc[:, 'y']])

    # showing the raw data without interpretation
    def present_raw_data(self):
        plt.figure(1)
        T, = plt.plot(self.target[0, :])
        F1, = plt.plot(self.feature1[0, :])
        F2, = plt.plot(self.feature2[0, :])
        F3, = plt.plot(self.feature3[0, :])
        F4, = plt.plot(self.feature4[0, :])
        plt.legend([T, F1, F2, F3, F4], (self.DATA))
        plt.show(block=False)

    def prep_data(self):
        self.X = np.concatenate([self.feature1, self.feature2, self.feature3, self.feature4])
        self.X = np.transpose(self.X)

        self.Y = self.target
        self.Y = np.transpose(self.Y)

        scaler = MinMaxScaler()
        scaler.fit(self.X)
        self.X = scaler.transform(self.X)
        self.X = np.reshape(self.X, (self.X.shape[0], 1, self.X.shape[1]))

        scaler1 = MinMaxScaler()
        scaler1.fit(self.Y)
        self.Y = scaler1.transform(self.Y)

    def split_train_test(self):
        l = train_test_split(self.X, self.Y, test_size=0.2)
        self.X_train = l[0]
        self.X_test = l[1]
        self.Y_train = l[2]
        self.Y_test = l[3]

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(100, activation='tanh', input_shape=(1, 4), recurrent_activation='hard_sigmoid'))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='rmsprop', metrics=[metrics.mae])
        self.model.fit(self.X_train, self.Y_train, epochs=100, verbose=2)

    def test_model(self):
        self.Predict = self.model.predict(self.X_test)
        plt.figure(2)
        plt.scatter(self.Predict, self.Y_test)
        plt.show(block=False)

        plt.figure(3)
        Test, = plt.plot(self.Y_test)
        Predict, = plt.plot(self.Predict)
        plt.legend([Predict, Test], ["Predicted Data", "Real Data"])
        plt.show()

m = MyModel()
m.fetch_data('data\\HK_anomaly_17_3\\HK')
m.present_raw_data()
m.prep_data()
m.split_train_test()
m.build_model()
m.test_model()
