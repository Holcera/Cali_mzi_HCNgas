from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hvplot.pandas
import holoviews as hv
import pyLPD.MLtools as pmlt
from pyparsing import alphas
import scipy
from bokeh.plotting import show
from scipy import signal, optimize, signal
from scipy import constants as C
# pd.options.plotting.backend = 'holoviews'

#%%
filepath = 'D:/wh/Documents/Lab/Data/DAQ_capture/daq1.5_10.7.csv'
data_raw = pd.read_csv(filepath, 
                       header=3, 
                       na_values='--',
                  #      skiprows=10000000,
                  #      nrows=1000000,
                       index_col=False,
                       usecols=[1, 3]
                       )
# data_raw.rename(columns={'CH1':'MZI', 'CH2':'trans'}, inplace=True)
data_raw.rename(columns={'V':'MZI', 'V.1':'trans'}, inplace=True)
data_raw.reset_index(inplace=True)

print('Memory usage:\n' + 
      str(round(data_raw.shape[1]*
                data_raw.memory_usage(index=False).mean()/1e6, 1)) + 'MB')
print(data_raw.head())
#%%
# figure1 = data_raw.hvplot(x='TIME', 
#                           y='trans', 
#                           kind='line',
#                           height=600, width=1300,
#                           alpha=0.8
#                           )
# show(hv.render(figure1))

# ax1 = data_raw.plot(x=data_raw.columns[0],
#                     y=data_raw.columns[6],
#                     kind='scatter',
#                     figsize=(12, 5)
#                     )

plt.figure(figsize=(15, 3))

plt.plot(data_raw.index, data_raw.MZI, c='r', lw=1, alpha=0.7, label='gas')
plt.plot(data_raw.index, data_raw.trans, c='b', lw=1, alpha=0.7, label='trans')


plt.show()

# %%
