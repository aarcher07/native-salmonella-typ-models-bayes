"""
Testing wild type model with a spline function trained on OD data.

Programme written by aarcher07
Editing History: See github history
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

GC_ODs_N = pd.read_excel("data/bobik_times_series_data_cleaned.xlsx", engine='openpyxl',header=[0,1]).dropna()

Time = GC_ODs_N.loc[:,('Time','Time (hrs)')].astype(np.float64)
OD_bMCPs = GC_ODs_N.loc[:,('Broken MCPs','OD')].astype(np.float64)
OD_MCPs = GC_ODs_N.loc[:,('WT','OD')].astype(np.float64)

# log transform and fit

# # Taken from https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
def sigmoid(x, L ,x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return y

p0 = [max(OD_bMCPs), np.median(Time), 1]  # this is an mandatory initial guess
popt, pcov = curve_fit(sigmoid, Time, OD_bMCPs, p0, method='dogbox')

fit_fun = lambda t: sigmoid(t, *popt)

# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] +2, num=int(1e3))
OD_bMCPs_spl = fit_fun(t)
plt.scatter(Time, OD_bMCPs)
plt.plot(t, OD_bMCPs_spl)
plt.legend(['data', 'Sigmoid'], loc='upper left')
plt.title('OD data fit to sigmoid for broken MCPs')
plt.savefig('od_plots/bobik_bMCP_sigmoid_OD_fit.png', bbox_inches="tight")
plt.close()

p0 = [max(OD_MCPs), np.median(Time), 1]  # this is an mandatory initial guess
popt, pcov = curve_fit(sigmoid, Time, OD_MCPs, p0, method='dogbox')

fit_fun = lambda t: sigmoid(t, *popt)

# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] +2, num=int(1e3))
OD_MCPs_spl = fit_fun(t)
plt.scatter(Time, OD_MCPs)
plt.plot(t, OD_MCPs_spl)
plt.legend(['data', 'Sigmoid'], loc='upper left')
plt.title('OD data fit to sigmoid for WT')
plt.savefig('od_plots/bobik_MCP_sigmoid_OD_fit.png', bbox_inches="tight")
plt.close()

# plot data and spline
plt.scatter(Time, OD_MCPs, label= 'growth data for WT')
plt.plot(t, OD_MCPs_spl, label = 'spline fit for WT')
plt.scatter(Time, OD_bMCPs, label = 'growth data for broken MCPs')
plt.plot(t, OD_bMCPs_spl, label = 'spline fit for broken MCPs')
plt.title('OD data fit to cubic spline')
plt.legend(loc='upper left')
plt.title('OD data fit to sigmoid')
plt.savefig('od_plots/bobik_sigmoid_OD_fit.png', bbox_inches="tight")
plt.close()
