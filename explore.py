import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PATH = 'data\\Kobi_Bryant_26_1-29_1\\AM' #path to the relevant data for explora

DATA = ('recommendation_requests_5m_rate_dc' , 
'total_failed_action_conversions' , 
'total_success_action_conversions' , 
'trc_requests_timer_p95_weighted_dc' , 
'trc_requests_timer_p99_weighted_dc')
DATE='2020-01-01' # year-month-day

FILE_NAMES = [x +'_'+DATE+'.csv' for x in DATA]
df = [pd.read_csv(PATH+'\\'+DATA[i]+'\\'+FILE_NAMES[i]) for i in range(0,len(DATA))]
X = [np.array(x.loc[:,'ds']) for x in df]

arrays = [np.array(x.loc[:,'y']) for x in df]

fig, axs = plt.subplots(len(DATA),1,sharex=True)
#Remove horisontal space between axes
fig.subplots_adjust(hspace=0)

# Plot each graph
for i in range(0,len(DATA)):
	axs[i].plot(X[i],arrays[i])
	axs[i].title.set_text(DATA[i])

plt.show()


