"""
Testing wild type model with a spline function trained on OD data.

Programme written by aarcher07
Editing History: See github history
"""

from wild_type_model import WildType
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from csaps import csaps
from constants import HRS_TO_SECS, OD_TO_COUNT_CONC

GC_ODs_N = pd.read_excel("data/bobik_times_series_data_cleaned.xlsx", engine='openpyxl',header=[0,1]).dropna()

Time = GC_ODs_N.loc[:,('Time','Time (hrs)')].astype(np.float64)
OD_bMCPs = GC_ODs_N.loc[:,('Broken MCPs','OD')].astype(np.float64)
OD_MCPs = GC_ODs_N.loc[:,('WT','OD')].astype(np.float64)

# plot data and spline for broken MCPs
spl = csaps(Time, OD_bMCPs, smooth=0.015)
t = np.linspace(0, Time.iloc[-1] + 2, num=int(1e2))
OD_bMCPs_spl = spl(t)
plt.scatter(Time, OD_bMCPs)
plt.plot(t, OD_bMCPs_spl)
plt.legend(['spline', 'data'], loc='upper left')
plt.title('OD data fit to cubic spline for broken MCPs')
plt.savefig('figures/bobik_bMCPs_spline_OD_fit.png', bbox_inches="tight")
plt.close()

# plot data and spline for WT
spl = csaps(Time, OD_MCPs, smooth=0.01)
OD_MCPs_spl = spl(t)
plt.scatter(Time, OD_MCPs)
plt.plot(t, OD_MCPs_spl)
plt.title('OD data fit to cubic spline for WT')
plt.legend(['spline', 'data'], loc='upper left')
plt.savefig('figures/bobik_MCP_spline_OD_fit.png', bbox_inches="tight")
plt.close()

# plot data and spline for WT
plt.scatter(Time, OD_MCPs, label= 'growth data for WT')
plt.plot(t, OD_MCPs_spl, label = 'spline fit for WT')
plt.scatter(Time, OD_bMCPs, label = 'growth data for broken MCPs')
plt.plot(t, OD_bMCPs_spl, label = 'spline fit for broken MCPs')
plt.title('OD data fit to cubic spline')
plt.legend(loc='upper left')
plt.savefig('figures/bobik_spline_OD_fit.png', bbox_inches="tight")
plt.close()
