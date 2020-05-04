import os
import pandas as pd
import numpy as np
import csv

from functools import reduce
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras import backend

from sklearn import preprocessing
import argparse


class Reader:
    # path is the path to the Main dataset folder
    def __init__(self, path):
        self.path = path

    # set the path again if you want to use the same reader to read another dataset
    def set_path(self, path):
        self.path = path

    # import all the data to one dataset
    def import_data(self):
        dirs = [x[0] for x in os.walk(self.path)]
        dirs.remove(self.path)
        self.dir_names = [dir.replace(self.path + os.path.sep, '') for dir in dirs]
        data_frames = []
        for dir in dirs:
            df = pd.concat(
                [pd.read_csv(os.path.join(dir, x)) for x in os.listdir(dir) if os.path.isfile(os.path.join(dir, x))])
            df.columns = ['date', dir.replace(self.path + os.path.sep, '')]
            data_frames.append(df)
        # merge
        dataset = reduce(lambda left, right: pd.merge(left, right, on='date'), data_frames)
        dataset.drop_duplicates(subset=None, inplace=True)
        dataset.drop('date', 1)
        dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
        return dataset


class Model:
    def __init__(self):
        pass

    def fetch_data(self, path):
        r = Reader(path)
        self.raw_dataset = r.import_data()
        self.feacher_names = r.dir_names

    def present_data(self, dataset, figure):
        plt.figure(figure)
        ax = plt.gca()
        names = list(dataset.columns)
        [dataset.plot(kind='line', y=y, ax=ax) for y in names]
        plt.show()

    def normalize_data(self):
        x = self.raw_dataset.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self.normalized_dataset = pd.DataFrame(x_scaled)
        self.normalized_dataset.columns = self.raw_dataset.columns

    def reshape_data(self, prediction, ignore, data_point_to_predict=0):
        if (ignore != None):
            self.feacher_names.remove(ignore)
        if (data_point_to_predict == 0):
            self.feacher_names.remove(prediction)
            pred = self.normalized_dataset[prediction].values
            self.prediction = pred.reshape(pred.shape[0], 1)
            features = self.normalized_dataset[self.feacher_names].values
            self.features = features.reshape(features.shape[0], 1, features.shape[1])
        else:
            pred = self.normalized_dataset[prediction].copy(deep=True).values
            print(pred)
            pred = pred[slice(data_point_to_predict, None)]
            print(pred)
            self.prediction = pred.reshape(pred.shape[0], 1)
            features = self.normalized_dataset[self.feacher_names].values
            features = features[slice(None, features.shape[0] - data_point_to_predict)]
            self.features = features.reshape(features.shape[0], 1, features.shape[1])

    def split_train_test(self, test_size, validation_size=0.1):
        relative_val_size = (validation_size / (1 - test_size))  # to make it allways equels to 10% of the data
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.features, self.prediction,
                                                                                test_size=test_size, shuffle=False)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train,
                                                                              test_size=relative_val_size,
                                                                              shuffle=False)

    def rme(self, y_true, y_pred):
        return backend.sqrt(abs(backend.mean(backend.square(y_pred - y_true), axis=-1)))

    def build_model(self, Nodes=100, LSTM_activation='relu', recurrent_activation='sigmoid', dense_activation='tanh',
                    optimizer='adam'):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(Nodes, input_shape=(self.features.shape[1], self.features.shape[2]))))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.compile(loss=self.rme, optimizer=optimizer, metrics=['mse', 'mae'])

    def train_model(self, epochs=30):
        history = self.model.fit(self.X_train, self.Y_train, epochs=epochs, verbose=1,
                                 validation_data=(self.X_val, self.Y_val))
        return (history.history)

    def test_model(self, test_size=0.1):
        self.Predict = self.model.predict(self.X_test)
        score = self.model.evaluate(self.X_test, self.Y_test)
        names = self.model.metrics_names
        score_dic = {}
        for i in range(0, len(names)):
            score_dic[names[i]] = score[i]
        plt.figure(2)
        plt.scatter(self.Predict, self.Y_test)
        plt.show(block=False)

        plt.figure(3)
        Test, = plt.plot(self.Y_test)
        Predict, = plt.plot(self.Predict)
        plt.legend([Test, Predict], ["Real Data", "Predicted Data"])
        plt.title('test size =' + str(test_size))
        plt.show(block=False)

        fig, (ax1, ax2) = plt.subplots(2, sharey=True)
        ax1.plot(self.Y_test)
        ax1.set(title="Real Data")
        ax2.plot(self.Predict)
        ax2.set(title="Predicted data")
        plt.show()
        return (score_dic)


def run(args):
    m = Model()
    m.fetch_data(args.path)
    m.present_data(m.raw_dataset, 1)
    m.normalize_data()
    m.present_data(m.normalized_dataset, 2)
    m.reshape_data(args.prediction, args.ignore, args.data_point_to_predict)
    m.split_train_test(args.test_size)
    m.build_model()
    m.train_model()
    m.test_model()


def evaluate(args):
    test_size_arr = np.linspace(0.1, 0.9, 8, endpoint=False)
    m = Model()
    m.fetch_data(args.path)
    m.normalize_data()
    m.reshape_data(args.prediction, args.ignore, args.data_point_to_predict)
    print(args.path)
    name_of_file = args.path.replace(os.path.sep + 'data', '')
    name_of_file = name_of_file.replace(os.path.sep, '-')
    with open('evaluation/' + name_of_file + '.csv', "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['test_size', 'val_loss', 'val_mse', 'val_mae', 'score_loss', 'score_mse', 'score_mae'])
        for test_size in test_size_arr:
            m.split_train_test(test_size)
            m.build_model()
            history = m.train_model()
            score = m.test_model(test_size)
            writer.writerow(
                [test_size, history['val_loss'][29], history['val_mse'][29], history['val_mae'][29], score['loss'],
                 score['mse'], score['mae']])
        csv_file.close()


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='This is an LSTM model to detect anomalies in data for Taboola')
    parser.add_argument('-path', action='store', dest='path')
    parser.add_argument('-prediction', action='store', dest='prediction')
    parser.add_argument('-test_size', action='store', dest='test_size', type=float)
    parser.add_argument('-ignore', action='store', dest='ignore', default=None)
    parser.add_argument('-predict_amount', action='store', dest='data_point_to_predict', type=int, default=0)
    args = parser.parse_args()
    run(args)
