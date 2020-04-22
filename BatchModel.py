import os
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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
        self.dir_names = [dir.replace(self.path + '\\', '') for dir in dirs]
        data_frames = []
        for dir in dirs:
            df = pd.concat(
                [pd.read_csv(os.path.join(dir, x)) for x in os.listdir(dir) if os.path.isfile(os.path.join(dir, x))])
            df.columns = ['date', dir.replace(self.path + '\\', '')]
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

    def reshape_data(self, prediction):
        pred = self.normalized_dataset[prediction].values
        self.prediction = pred.reshape(pred.shape[0], 1)
        self.feacher_names.remove(prediction)
        feachers = self.normalized_dataset[self.feacher_names].values
        self.feachers = feachers.reshape(feachers.shape[0], 1, feachers.shape[1])

    def split_train_test(self, test_size):
        print(self.feachers.shape)
        print(self.prediction.shape)
        l = train_test_split(self.feachers, self.prediction, test_size=test_size, shuffle=False)
        self.X_train = l[0]
        self.X_test = l[1]
        self.Y_train = l[2]
        self.Y_test = l[3]

    def rme(self, y_true, y_pred):
        return backend.sqrt(abs(backend.mean(backend.square(y_pred - y_true), axis=-1)))

    def build_model(self, Nodes=50, LSTM_activation='relu', recurrent_activation='sigmoid', dense_activation='tanh',
                    optimizer='adam'):
        self.model = Sequential()
        self.model.add(
            LSTM(Nodes,input_shape=(self.feachers.shape[1], self.feachers.shape[2])))
        self.model.add(Dense(1))
        self.model.compile(loss=self.rme, optimizer=optimizer, metrics=['mse', 'mae', 'mape', 'cosine'])

    def train_model(self, epochs=30):
        self.model.fit(self.X_train, self.Y_train, epochs=epochs, verbose=1)

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


def run(args):
    m = Model()
    m.fetch_data(args.path)
    m.present_data(m.raw_dataset, 1)
    m.normalize_data()
    m.present_data(m.normalized_dataset, 2)
    m.reshape_data(args.prediction)
    m.split_train_test(0.2)
    m.build_model()
    m.train_model()
    m.test_model()


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='This is an LSTM model to detect anomalies in data for Taboola')
    parser.add_argument('-path', action='store', dest='path')
    parser.add_argument('-prediction', action='store', dest='prediction')
    args = parser.parse_args()
    print(args.path)
    run(args)
