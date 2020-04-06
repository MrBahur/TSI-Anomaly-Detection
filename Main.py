import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import glob

import argparse


class Reader:
    # path is the path to the Main dataset folder
    def __init__(self, path):
        self.path = path

    # set the path again if you want to use the same reader to read another dataset
    def set_path(self, path):
        self.path = path

    # read all the csv file in self.path
    # return Data Frame containing all of the data
    # TODO add read from date to date (in a range) another method
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

    # TODO get DATA as parameter and index for "target" data
    # where target is the data you want to predict
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
        self.feature = []
        for i in range(1, len(self.DATA)):
            reader.set_path(path=path + '//' + self.DATA[i])
            self.feature.append(np.array([reader.read().loc[:, 'y']]))

    # showing the raw data without interpretation
    # TODO find out how to make it in loop
    def present_raw_data(self):
        plt.figure(1)
        T, = plt.plot(self.target[0, :])
        F1, = plt.plot(self.feature[1 - 1][0, :])
        F2, = plt.plot(self.feature[2 - 1][0, :])
        F3, = plt.plot(self.feature[3 - 1][0, :])
        F4, = plt.plot(self.feature[4 - 1][0, :])
        plt.legend([T, F1, F2, F3, F4], (self.DATA))
        plt.show(block=False)

    # preparing the data (Min max normalization and shape of the data)
    def concatenate_data(self):
        self.X = np.concatenate(self.feature)
        self.X = np.transpose(self.X)

        self.Y = self.target
        self.Y = np.transpose(self.Y)

    def normalize(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(self.X)
        self.X = scaler.transform(self.X)

        scaler1 = MinMaxScaler(feature_range=(0, 1))
        scaler1.fit(self.Y)
        self.Y = scaler1.transform(self.Y)

    def reshape_X(self):
        self.X = np.reshape(self.X, (self.X.shape[0], 1, self.X.shape[1]))
        print(self.X.shape)

    def prep_data(self):
        self.concatenate_data()
        self.normalize()
        self.reshape_X()

    # split the data to train and test (as a batch)
    def split_train_test(self):
        l = train_test_split(self.X, self.Y, test_size=0.2)
        self.X_train = l[0]
        self.X_test = l[1]
        self.Y_train = l[2]
        self.Y_test = l[3]

    # building and fitting the model
    def build_model(self, Nodes=50, LSTM_activation='tanh', recurrent_activation='hard_sigmoid',
                    dense_activation='sigmoid', num_of_layers=1):
        self.model = Sequential()
        for i in range(1, num_of_layers):
            self.model.add(
                LSTM(Nodes, activation=LSTM_activation, input_shape=(self.X.shape[1],self.X.shape[2]),
                     recurrent_activation=recurrent_activation, return_sequences=True))
        self.model.add(
            LSTM(Nodes, activation=LSTM_activation, input_shape=(self.X.shape[1],self.X.shape[2]),
                 recurrent_activation=recurrent_activation))
        self.model.add(Dense(1, activation=dense_activation))
        self.model.compile(loss='mse', optimizer='adam', metrics=[metrics.mae])

    def train_model(self, epochs=30):
        self.model.fit(self.X_train, self.Y_train, epochs=epochs, verbose=2, )

    # testing the model
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


def main(args=None):
    m = MyModel()
    m.fetch_data(args.path)
    m.present_raw_data()
    m.prep_data()
    m.split_train_test()
    m.build_model()
    m.train_model()
    m.test_model()


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='This is an LSTM model to detect anomalies in data for Taboola')
    parser.add_argument('-path', action='store', dest='path')
    args = parser.parse_args()
    print(args.path)
    main(args)
