# -*- coding: utf-8 -*-
"""
Created on Sun May 22 14:33:20 2022

@author: Heeeg
"""
from cProfile import label
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hvplot.pandas
import holoviews as hv
import pyLPD.MLtools as pmlt
from pyparsing import alphas
import scipy
from bokeh.plotting import show
from scipy import signal, optimize, signal, interpolate
from scipy import constants as C
from matplotlib import rcParams
# pd.options.plotting.backend = 'holoviews'
#%%
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 20
#%%
filepath = r"C:\Users\H\Documents\Lab_BUPT\Data\DAQ_data\MZI\DAQ1.csv"
data_raw = pd.read_csv(filepath, 
                       header=3, 
                       na_values='--',
                       usecols=[1, 3, 5]
                       )
data_raw.rename(columns={'V':'MZI', 'V.1':'gas', 'V.2':'trans'}, inplace=True)
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

ax1 = data_raw.plot(x='index',
                    y='trans',
                    kind='line',
                    figsize=(12, 5)
                    )
plt.show()
#%%
t_start = 2.39e6
t_end = 1.507e7

t_start = data_raw[data_raw.index == t_start].index.tolist()[0]
t_end = data_raw[data_raw.index == t_end].index.tolist()[0]
data_cut = data_raw.iloc[t_start:t_end, :].copy()
# print(data_cut)
data_cut.reset_index(drop=True, inplace=True)
data_cut.drop(['index'], axis=1, inplace=True)

print('Memory usage:\n' + 
      str(round(data_cut.shape[1]*
                data_cut.memory_usage(index=False).mean()/1e6, 1)) + 'MB')
print(data_cut)
#%%
plt.figure(figsize=(15, 3))
plt.plot(data_cut.index, data_cut.gas, c='r', lw=1, alpha=0.7, label='gas')
plt.plot(data_cut.index, data_cut.trans*10, c='b', lw=1, alpha=0.7, label='trans')
plt.plot(data_cut.index, data_cut.MZI, c='purple', label='mzi')
plt.xlim(data_cut.index[int(8e3)], data_cut.index[int(1e4)])

plt.show()
#%%
## nor mzi

data_cut['mzi_min'], data_cut['mzi_max'] = pmlt.envPeak(data_cut.MZI.values, delta=0.15, sg_order=0)
data_cut['mzi_nor'] = (data_cut.MZI-data_cut.mzi_min) / (data_cut.mzi_max-data_cut.mzi_min)
data_cut['mzi_sm'] = scipy.signal.savgol_filter(data_cut.mzi_nor.values, 21, 1)
# print(data_cut.head())
#%%
delta = 0.4
ind_max, maxtab, ind_min, mintab = pmlt.peakdet(data_cut.mzi_sm.values, delta)
ind_peaks = np.sort(np.concatenate((ind_min, ind_max), axis=0))
print(ind_peaks)
#%%
plt.figure(figsize=(15, 3))
plt.subplot(211)
plt.scatter(data_cut.index, data_cut.mzi_nor, c='g', s=5, label='mzi_data')
plt.plot(data_cut.index, data_cut.mzi_sm, c='r', lw=1, alpha=0.7, label='mzi_sm')
plt.xlim(data_cut.index[int(1e3)], data_cut.index[int(3e3)])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.subplot(212)
plt.plot(data_cut.index, data_cut.mzi_sm, c='r', lw=1, alpha=0.7, label='mzi_sm')
plt.scatter(data_cut.index[ind_min], mintab, s=5, c='purple', alpha=0.5, label='min')
plt.scatter(data_cut.index[ind_max], maxtab, s=5, c='royalblue', alpha=0.5, label='max')
plt.xlim(data_cut.index[int(2e3)], data_cut.index[int(3e3)])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
#%%
plt.figure(figsize=(18, 7))
plt.scatter(data_cut.index, data_cut.mzi_nor, c='g', s=5, label='MZI Exp.')
plt.scatter(data_cut.index[ind_min], mintab, s=50, c='darkred', label='Minimum')
plt.scatter(data_cut.index[ind_max], maxtab, s=50, c='darkred', label='maximum')
plt.plot(data_cut.index, data_cut.trans*5, c='slateblue', lw=1, alpha=0.7, label='Cavity Exp.')
plt.plot(data_cut.index, data_cut.mzi_sm, c='r', lw=1, alpha=1, label='Fitting')
plt.xlim(data_cut.index[int(8e5)], data_cut.index[int(8.1e5)])
plt.ylim(min(data_cut.mzi_nor), max(data_cut.trans*5)-0.5)
plt.legend(loc='center left', fontsize=19, frameon=False)
plt.ylabel('Intensity (a.u.)')
plt.xlabel('Frequency (MHz)')
plt.yticks([]) 
plt.xticks([]) 

plt.show()
#%%
## nor hcn gas cell

data_cut['low_gas'], data_cut['up_gas'] = pmlt.envPeak(data_cut.gas.values, delta=0.12, smooth=0.12, sg_order=2)
data_cut['gas_n'] = data_cut.gas / (data_cut['up_gas'])

# data_cut['gas_n'] = (data_cut.gas - data_cut.gas.min()) / (data_cut.gas.max() - data_cut.gas.min())
ind_max_gas, maxtab_gas, ind_min_gas, mintab_gas = pmlt.peakdet(data_cut.gas_n.values, 0.12)

plt.figure(figsize=(21, 4))
plt.plot(data_cut.index[ind_min_gas[0]:ind_min_gas[-1]], data_cut.gas_n[ind_min_gas[0]:ind_min_gas[-1]], 
         label='HCN gas cell'
         )
plt.scatter(data_cut.index[ind_min_gas], mintab_gas, c='purple', s=50, label='min')

ax = plt.gca()
for i in range(0, len(ind_min_gas)):
      ax.annotate(i, (data_cut.index[ind_min_gas[i]], mintab_gas[i]), color='r', fontsize=15)

plt.xlabel('Times')
plt.ylabel('Gas Trans')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

#%%
peak_gas = 35
lambda35 = 1548.95555 #nm
freq35 = 1e-3 * C.c / lambda35

print(f'λ35 = {lambda35} nm \nfreq35 = {round(freq35, 6)} THz')
#%%
gas_lambda = [1527.63342, 1528.05474, 1528.48574, 1528.92643, 1529.37681, 
              1529.83688, 1530.30666, 1530.78615, 1531.27537, 1531.77430, 
              1532.28298, 1532.80139, 1533.32954, 1533.86745, 1534.41514, 
              1534.97258, 1535.53981, 1536.11683, 1536.70364, 1537.30029, 
              1537.90675, 1538.52305, 1539.14921, 1539.78523, 1540.43120, 
              1541.08703, 1541.75280, 1543.11423, 1543.80967, 1544.51503, 
              1545.23033, 1545.95549, 1546.69055, 1547.43558, 1548.19057,
              1548.95555, 1549.73051, 1550.51546, 1551.31045, 1552.11546,
              1552.93051, 1553.75562, 1554.59079, 1555.43605, 1556.29141,
              1557.15686, 1558.03240, 1558.91808, 1559.81389, 1560.71983,
              1561.63593, 1562.56218, 1563.49859, 1564.44519]

gas_freq = 1e-3 * C.c / np.array(gas_lambda) #THz
data1 = {'gas_lambda':gas_lambda,
         'gas_freq':gas_freq
        }
data_gas = pd.DataFrame(data1)

print(data_gas.head())
#%%
freq_interfunc = scipy.interpolate.interp1d(data_cut.index[ind_min_gas], data_gas.gas_freq)
data_icut = data_cut.iloc[min(ind_min_gas):max(ind_min_gas),:].copy()
data_icut['ifreq'] = freq_interfunc(data_icut.index)
data_icut.reset_index(drop=True, inplace=True)

ind_max_icut, maxtab_icut, ind_min_icut, mintab_icut = pmlt.peakdet(data_icut.mzi_sm.values, delta=0.4)
ind_peaks_icut = np.sort(np.concatenate((ind_min_icut, ind_max_icut), axis=0))
# plt.figure(figsize=(15, 3))
# # plt.scatter(data_icut.TIME, data_icut.mzi_nor, c='g', lw=0.5)
# plt.plot(data_icut.TIME, data_icut.mzi_sm, c='r', lw=1, alpha=0.7, label='mzi_sm')
# plt.scatter(data_icut.TIME[ind_min_icut], mintab_icut, s=10, c='r', alpha=0.5, label='min')
# plt.scatter(data_icut.TIME[ind_max_icut], maxtab_icut, s=10, c='r', alpha=0.5, label='max')
# plt.xlim(data_icut.TIME[0], data_icut.TIME[2e3])

#%%
ind_cen = round(len(ind_peaks_icut)/2)
range_mzi = (np.arange(len(ind_peaks_icut)) - ind_cen)

print(data_icut)
print(len(range_mzi), len(ind_peaks_icut))
#%%
def dispersion_fit(mu, *p):
      return freq_cen - p[0]*mu - p[1]/2*(mu**2) - p[2]/6*(mu**3)
#%%
cen1 = ind_peaks_icut[ind_cen]
freq_cen = data_icut.ifreq[cen1] #THz
# print(data_icut.ifreq[])
pfit_disp, pcov_disp = optimize.curve_fit(dispersion_fit, range_mzi, data_icut.ifreq[ind_peaks_icut], [0,0,0])

print('MZI Parameters: \n' + 'D1_mzi = {:.5g} MHz, D2_mzi = {:.4g} Hz, D3_mzi = {:.3g} μHz'.format(1e6*pfit_disp[0]*2, pfit_disp[1]*1e12*2, pfit_disp[2]*1e18*2))
# print('MZI Parameters: \n' + 'D1_mzi = {:.5g} MHz, D2_mzi = {:.4g} Hz, D3_mzi = {:.3g} μHz'.format(1e6*pfit_disp[0], pfit_disp[1]*1e12, pfit_disp[2]*1e18))
#%%
plt.figure(figsize=(4, 3))
plt.scatter(range_mzi, data_icut.ifreq[ind_peaks_icut]-freq_cen, c='royalblue', s=5, alpha=0.5, label='Data')
plt.plot(range_mzi, dispersion_fit(range_mzi, *pfit_disp)-freq_cen, c='r', label='Fit')
plt.xlabel('Mode number (n)')
plt.ylabel('fn - f0 (THz)')
plt.legend(loc=7)

plt.show()
#%%
## Check gas cell wavelenth

# range_mzic = range_mzi / 2
# freq_r = freq35 - pfit_disp[0]*range_mzic - pfit_disp[1]/2*range_mzic**2 - pfit_disp[2]/6*range_mzic**3
# freq_ifunc_check = scipy.interpolate.interp1d(data_cut.TIME[ind_peaks], freq_r)

# data_checut = data_cut.iloc[min(ind_peaks):max(ind_peaks), :].copy()
# data_checut['freq_che'] = freq_ifunc_check(data_icut.TIME)
# data_checut.reset_index(drop=True, inplace=True)

# plt.figure(figsize=(25,5))
# plt.plot(data_checut.freq_che[ind_min_gas[0]:ind_min_gas[-1]], data_checut.gas_n[ind_min_gas[0]:ind_min_gas[-1]])
# plt.scatter(data_checut.freq_che[ind_min_gas], mintab_gas)
# plt.title('check the wavelenth')
# colors = itertools.cycle(['r', 'g', 'b'])

# ax = plt.gca()
# for ii in range(0,len(ind_min_gas)):
#     ax.annotate('{:.1f} nm'.format(1e-3*C.c/data_checut.freq_che[ind_min_gas[ii]]), (data_checut.freq_che[ind_min_gas[ii]], mintab_gas[ii]), color=next(colors), rotation=70)

# plt.grid(True)
# plt.xlabel('Frequency (THz)')
# plt.ylabel('Norm. Trans.')
# plt.show()
# %%
# data_save = pd.DataFrame()
# data_save['freq'] = data_icut.ifreq
# data_save['mzi'] = data_icut.mzi_sm
# data_save['trans'] = data_icut.trans

# data_save.to_csv('D:/wh/Documents/py_works/Cavity_disper/data_daq1.csv', encoding='utf-8')
#%%