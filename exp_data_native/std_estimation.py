"""
Computes the standard deviation of the experimental exp_data in mM for glycerol and PDO, and
g/L for DCW and stores a dictionary
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, abspath
import pandas as pd
import pickle
ROOT_PATH = dirname(dirname(abspath(__file__)))

time_series_df_cleaned_complete = pd.read_excel(ROOT_PATH + "/exp_data_native/data_files/bobik_times_series_data_cleaned.xlsx", engine='openpyxl',header=[0,1]).dropna()
time_series_df_cleaned_complete_sub = time_series_df_cleaned_complete.loc[:, 'WT']
time_samples = time_series_df_cleaned_complete.loc[:,('Time','Time (hrs)')].astype(np.float64)

sigma_all_exp= {}



for exp_cond in ['Broken MCPs', 'WT']:
    time_series_df_cleaned_complete_sub = time_series_df_cleaned_complete.loc[:, exp_cond]
    sigma_data = []
    for reactant in time_series_df_cleaned_complete_sub.columns[1:]:
        sigma_sq_data = []
        data_list = []
        data_samples = time_series_df_cleaned_complete_sub.loc[:, reactant]
        if exp_cond == 'WT':
            if reactant == "12PD (mM)":
                sigma_sq_val = 0.5
                mass_1 = 1.75
                mass_2 = 1e-2
            elif reactant == "1-Propanol (mM)":
                sigma_sq_val = 0.25
                mass_1 = 1.75
                mass_2 = 1e-2
            elif reactant == 'Propionate (mM)':
                sigma_sq_val = 0.05
                mass_1 = 0.4375
                mass_2 = 1e-2
            else:
                sigma_sq_val = 0.0005
                mass_1 = 0.000001
                mass_2 = 1e-2

        if exp_cond == 'Broken MCPs':
            if reactant == "12PD (mM)":
                sigma_sq_val = 0.5
                mass_1 = 1.75
                mass_2 = 1e-2
            elif reactant == "1-Propanol (mM)":
                sigma_sq_val = 0.25
                mass_1 = 1.75
                mass_2 = 1e-2
            else:
                sigma_sq_val = 0.05
                mass_1 = 1
                mass_2 = 1e-2

        for i in data_samples.index:
            if data_samples.loc[i] > mass_1:
                sigma_sq = sigma_sq_val
            elif data_samples.loc[i] > mass_2:
                sigma_sq = (data_samples[i] / 3.5) ** 2
            else:
                sigma_sq = 1e-9
            sigma_sq_data.append(sigma_sq)
        # complete exp_data
        plt.errorbar(time_samples, data_samples, yerr=2*np.sqrt(np.array(sigma_sq_data)), fmt='none')
        plt.show()

        sigma_data.append(np.sqrt(sigma_sq_data).tolist())
    df = pd.DataFrame(sigma_data).transpose()
    df.columns= time_series_df_cleaned_complete_sub.columns[1:]
    sigma_all_exp[exp_cond] = df.to_numpy()

with open('data_files/std_experimental_data.pkl', 'wb') as f:
    pickle.dump(sigma_all_exp, f, pickle.HIGHEST_PROTOCOL)