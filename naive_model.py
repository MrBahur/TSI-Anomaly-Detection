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
        'total_success_action_conversions',
        'trc_requests_timer_p95_weighted_dc',
        'trc_requests_timer_p99_weighted_dc')

#collecting all the data from 2020-01-01 to 2020-25-01
arrays=[[] for i in range(0,len(DATA))]
Xs = [[] for i in range(0,len(DATA))]

for j in range(1,25):
	DATE = datetime.datetime(2020, 1, j).strftime("%Y-%m-%d") 
	FILE_NAMES = [x + '_' + DATE + '.csv' for x in DATA]
	data_types = {'ds': str, 'y': float}
	parse_dates = ['ds']
	dfs = [pd.read_csv(PATH + '\\' + DATA[i] + '\\' + FILE_NAMES[i], dtype=data_types, parse_dates=parse_dates,
                   date_parser=pd.to_datetime) for i in range(0, len(DATA))]
	for i in range(0,4):
		t1 = dfs[i].loc[:, 'y']
		t2 = dfs[i].loc[:,'ds']
		arrays[i].append(t1)
		Xs[i].append(t2)
data = [np.concatenate(x) for x in arrays] # data[0] = recommendation_requests_5m_rate_dc,
# data[1] = total_success_action_conversions, data[2] = trc_requests_timer_p95_weighted_dc, data[3] = trc_requests_timer_p99_weighted_dc
X = [np.concatenate(x) for x in Xs]

fig, axs = plt.subplots(len(DATA), 1, sharex=True)
fig.subplots_adjust(hspace=0.4)

for i in range(0, len(DATA)):
    axs[i].plot(X[i], data[i])
    axs[i].title.set_text(DATA[i])
    for label in axs[i].get_xticklabels():
        label.set_rotation(40)
        label.set_horizontalalignment('right')

plt.show()

