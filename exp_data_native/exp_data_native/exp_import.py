from os.path import dirname, abspath
import pandas as pd
import pickle
ROOT_PATH = dirname(dirname(abspath(__file__)))

# reformat experimental data as dictionary
experimental_time_series_df = pd.read_excel(ROOT_PATH + "/data_files/bobik_times_series_data_cleaned.xlsx",
                                            engine='openpyxl',header=[0,1]).dropna()

TIME_SAMPLES = {} # dictionary of time samples for each initial glycerol concentration experiment

TIME_SAMPLES_BOBIK = experimental_time_series_df.loc[:, ('Time', 'Time (hrs)')].to_numpy()
DATA_SAMPLES = {} # dictionary of 1,2-PD, Propionaldehyde, Propionate and 1-propanol for WT and broken MCPs

for exp_cond in ['Broken MCPs', 'WT']:
    DATA_SAMPLES[exp_cond] = experimental_time_series_df.loc[:,exp_cond].to_numpy()[:,1:]


# initial conditions
INIT_CONDS = {exp_cond:DATA_SAMPLES[exp_cond][0,:] for exp_cond in ['Broken MCPs', 'WT']}
with open(ROOT_PATH + '/data_files/std_experimental_data.pkl', 'rb') as f:
    STD_EXPERIMENTAL_DATA = pickle.load(f)

with open(ROOT_PATH + '/data_files/od_fit_params_bMCP_rat.pkl', 'rb') as f:
    OD_FIT_PARAMS_BROKEN_MCPS_RAT = pickle.load(f)

with open(ROOT_PATH + '/data_files/od_fit_params_MCP_sigmoid.pkl', 'rb') as f:
    OD_FIT_PARAMS_MCPS_SIGMOID = pickle.load(f)