import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


PATH = 'data\\Kobi_Bryant_26_1-29_1\\AM' #path to the relevant data for explora

DATA = ('recommendation_requests_5m_rate_dc' , 
'total_failed_action_conversions' , 
'total_success_action_conversions' , 
'trc_requests_timer_p95_weighted_dc' , 
'trc_requests_timer_p99_weighted_dc')
base = datetime.datetime(2020,1,26) #ser the date you want to explore year, day, month
DATE=base.strftime("%Y-%m-%d")

FILE_NAMES = [x +'_'+DATE+'.csv' for x in DATA]
dfs = [pd.read_csv(PATH+'\\'+DATA[i]+'\\'+FILE_NAMES[i]) for i in range(0,len(DATA))]
Xs = [np.array(df.loc[:,'ds']) for df in dfs]
arrays = [np.array(df.loc[:,'y']) for df in dfs]

fig, axs = plt.subplots(len(DATA),1,sharex=False)
#decrease horisontal space between axes
fig.subplots_adjust(hspace=.7)
fig.suptitle(DATE)
# Plot each graph
for i in range(0,len(DATA)):
	axs[i].plot(Xs[i],arrays[i])
	axs[i].title.set_text(DATA[i])
	[l.set_visible(False) for (j,l) in enumerate(axs[i].xaxis.get_ticklabels()) if j%12 !=0]
	for label in axs[i].get_xticklabels():
		label.set_rotation(40)
		label.set_horizontalalignment('right')

plt.show()


