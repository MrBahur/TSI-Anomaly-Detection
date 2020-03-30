import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
PATH = 'data\\Kobi_Bryant_26_1-29_1\\AM'  # path to the relevant data for explore

DATA = ('recommendation_requests_5m_rate_dc',
        'total_failed_action_conversions',
        'total_success_action_conversions',
        'trc_requests_timer_p95_weighted_dc',
        'trc_requests_timer_p99_weighted_dc')
DATE = datetime.datetime(2020, 1, 25).strftime("%Y-%m-%d") # set the date you want to explore year, day, month
