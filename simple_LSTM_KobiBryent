# multivariate lstm example
from numpy import array
from numpy import hstack
from numpy import loadtxt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import datetime

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


base = datetime.datetime(2020, 1, 1)  # set the date year, day, month
DATE = base.strftime("%Y-%m-%d")
#load the datasets
dataset_5M = pd.read_csv('recommendation_requests_5m_rate_dc_'+DATE+'.csv')
dataset_P95 = pd.read_csv('trc_requests_timer_p95_weighted_dc_'+DATE+'.csv')
dataset_P99 = pd.read_csv('trc_requests_timer_p99_weighted_dc_'+DATE+'.csv')
dataset_SuccessAction = pd.read_csv('total_success_action_conversions_'+DATE+'.csv')
# define input sequence
inSeq_5M = np.array(dataset_5M.loc[:,'y'])
inSeq_P95 = np.array(dataset_P95.loc[:,'y'])
inSeq_P99 = np.array(dataset_P99.loc[:,'y'])
out_seq_SuccessAction = np.array(dataset_SuccessAction.loc[:,'y'])
# convert to [rows, columns] structure
inSeq_5M = inSeq_5M.reshape(len(inSeq_5M), 1)
inSeq_P95 = inSeq_P95.reshape((len(inSeq_P95), 1))
inSeq_P99 = inSeq_P99.reshape((len(inSeq_P99), 1))
out_seq_SuccessAction = out_seq_SuccessAction.reshape((len(out_seq_SuccessAction), 1))
# horizontally stack columns
dataset = hstack((inSeq_5M, inSeq_P95, inSeq_P99, out_seq_SuccessAction))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
base = datetime.datetime(2020, 1, 2)  # set the date you want to prediction year, day, month
DATE = base.strftime("%Y-%m-%d")

hour, minute = (0,15) #set the time you want to prediction hour, minute (XX:XX)
minute = minute/5
hour = hour*12
time = (int)(minute+hour)
print (time)
#load the datasets
datasetIn_5M = pd.read_csv('recommendation_requests_5m_rate_dc_'+DATE+'.csv')
datasetIn_P95 = pd.read_csv('trc_requests_timer_p95_weighted_dc_'+DATE+'.csv')
datasetIn_P99 = pd.read_csv('trc_requests_timer_p99_weighted_dc_'+DATE+'.csv')
datasetOut_SuccessAction = pd.read_csv('total_success_action_conversions_'+DATE+'.csv')
# define input sequence
input_5M = np.array(datasetIn_5M.loc[:,'y'])
input_P95 = np.array(datasetIn_P95.loc[:,'y'])
input_P99 = np.array(datasetIn_P99.loc[:,'y'])
output_SuccessAction = np.array(datasetOut_SuccessAction.loc[:,'y'])

x_input = array([[input_5M[time-2], input_P95[time-2], input_P99[time-2]], [input_5M[time-1], input_P95[time-1], input_P99[time-1]], [input_5M[time], input_P95[time], input_P99[time]]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

print(' %d (expected %d)' % (yhat, output_SuccessAction[time]))
