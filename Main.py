import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
        self.target = reader.read()
        reader.set_path(path=path + '//' + self.DATA[1])
        self.feature1 = reader.read()
        reader.set_path(path=path + '//' + self.DATA[2])
        self.feature2 = reader.read()
        reader.set_path(path=path + '//' + self.DATA[3])
        self.feature3 = reader.read()
        reader.set_path(path=path + '//' + self.DATA[4])
        self.feature4 = reader.read()

    # showing the raw data without interpretation
    def present_raw_data(self):
        Target = np.array([self.target.loc[:, 'y']])
        Feature1 = np.array([self.feature1.loc[:, 'y']])
        Feature2 = np.array([self.feature2.loc[:, 'y']])
        Feature3 = np.array([self.feature3.loc[:, 'y']])
        Feature4 = np.array([self.feature4.loc[:, 'y']])

        plt.figure(1)
        T, = plt.plot(Target[0, :])
        F1, = plt.plot(Feature1[0, :])
        F2, = plt.plot(Feature2[0, :])
        F3, = plt.plot(Feature3[0, :])
        F4, = plt.plot(Feature4[0, :])
        plt.legend([T,F1,F2,F3,F4],(self.DATA))
        plt.show()


