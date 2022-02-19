import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from os.path import dirname, abspath
import pickle
ROOT_PATH = dirname(dirname(abspath(__file__)))

GC_ODs_N = pd.read_excel(ROOT_PATH + "/exp_data_native/data_files/bobik_times_series_data_cleaned.xlsx", engine='openpyxl',header=[0,1]).dropna()
Time = GC_ODs_N.loc[:,('Time','Time (hrs)')].astype(np.float64)
OD_bMCPs = GC_ODs_N.loc[:, ('Broken MCPs', 'OD')].astype(np.float64)
OD_MCPs = GC_ODs_N.loc[:,('WT','OD')].astype(np.float64)

def sigmoid(x, L ,x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return y

def rational(x, p, q):
    """
    The general rational function description.
    p is a list with the polynomial coefficients in the numerator
    q is a list with the polynomial coefficients (except the first one)
    in the denominator
    The zeroth order coefficient of the denominator polynomial is fixed at 1.
    Numpy stores coefficients in [x**2 + x + 1] order, so the fixed
    zeroth order denominator coefficent must comes last. (Edited.)
    """
    return np.polyval(p, x) / np.polyval(q + [1.0], x)

def rational3_3(x, p0, p1, p2, p3, q0,q1, q2):
    return rational(x, [p0, p1, p2, p3], [q0, q1, q2])

popt_bMCPs, pcov = curve_fit(rational3_3, Time, OD_bMCPs)
OD_bMCPs_rat = lambda t: rational3_3(t, *popt_bMCPs)

p0 = [max(OD_MCPs), np.median(Time), 1]  # this is an mandatory initial guess
popt_MCPs, pcov = curve_fit(sigmoid, Time, OD_MCPs, p0, method='dogbox')

OD_MCPs_sig = lambda t: sigmoid(t, *popt_MCPs)

# plot data and spline
t = np.linspace(0, Time.iloc[-1] +2, num=int(1e3))
plt.scatter(Time, OD_MCPs, label= 'growth data for WT')
plt.plot(t, OD_MCPs_sig(t), label = 'sigmoid fit for WT')
plt.scatter(Time, OD_bMCPs, label = 'growth data for broken MCPs')
plt.plot(t, OD_bMCPs_rat(t), label ='rational fit for broken MCPs')
plt.title('OD data fit')
plt.legend(loc='upper left')
plt.title('OD data fit to sigmoid')
plt.show()

with open(ROOT_PATH + "/exp_data_native/data_files/od_fit_params_MCP_sigmoid.pkl", 'wb') as f:
    pickle.dump(popt_MCPs, f, pickle.HIGHEST_PROTOCOL)
with open(ROOT_PATH + "/exp_data_native/data_files/od_fit_params_bMCP_rat.pkl", 'wb') as f:
    pickle.dump(popt_bMCPs, f, pickle.HIGHEST_PROTOCOL)