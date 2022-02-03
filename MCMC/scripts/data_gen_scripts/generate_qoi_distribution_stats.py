import os
from pdo_model_pytorch.misc_functions import load_obj, save_obj
from pdo_model_pytorch.model_constants import MODEL_PARAMETER_LIST,VARS_TO_TEX, SINGLE_EXP_CALIBRATION_LIST
from pdo_model_pytorch.data_set_constants import INIT_CONDS_GLY_PDO_DCW,INTERSECTION_TIME_EVALS,DATA_SAMPLES,TIME_SAMPLES
from MCMC.dhaB_dhaT_model_prior import DhaBDhaTModelMCMC
from MCMC.prior_constants import LOG_NORM_PRIOR_PARAMETERS, DATA_LOG_UNIF_PARAMETER_RANGES
from MCMC.dhaB_dhaT_model_bayesian_funs import loglik
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from os.path import dirname, abspath
import pandas as pd
import scipy.stats as stats
import matplotlib.ticker as ticker
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
from mpi4py import MPI
from pydlc import dense_lines
import matplotlib.colors as mcolors
import signal

ROOT_PATH = dirname(dirname(dirname(dirname(abspath(__file__)))))
burn_in_n = int(1e3)
skip = 10
ntimepts = 1000

# create filenames and load files names
directory_data_formatted = ROOT_PATH + "/MCMC/output/MCMC_results_formatted/adaptive/preset_std/lambda_0,01_beta_0,25/nsamples_10000/norm"
directory_plot = '/' + '/'.join([st if not st == "MCMC_results_formatted" else "MCMC_results_plots" for st in
                                 directory_data_formatted.split('/')]) + '/interpretation_plots'
data_dict = load_obj(directory_data_formatted + '/data_burnin_' + str(burn_in_n) + '_skip_' + str(int(skip)))

# reformat to array
parameter_samples_array = np.array([val for key, val in data_dict.items()]).T #TODO remove 0
median_parameter_samples_array = np.median(parameter_samples_array, axis=0)
median_param_dict = {param_name: param for param_name, param in
                     zip(SINGLE_EXP_CALIBRATION_LIST, median_parameter_samples_array)}

mean_parameter_samples_array = np.mean(parameter_samples_array, axis=0)
mean_param_dict = {param_name: param for param_name, param in
                   zip(SINGLE_EXP_CALIBRATION_LIST, mean_parameter_samples_array)}

# load DCW exp_data_native
DCW_time_data, DCW_data = load_obj(ROOT_PATH + '/exp_data_native/data_files/DCW_discretized_N_100')

# create model
dhaB_dhaT_model = DhaBDhaTModelMCMC(transform="log_norm")
for gly_cond in TIME_SAMPLES.keys():
    data_sample_df = DATA_SAMPLES[gly_cond]  # experimental exp_data_native
    tsamp = TIME_SAMPLES[gly_cond]
    teval = np.linspace(0, tsamp[-1], ntimepts)
    init_conds = {'G_CYTO_INIT': 0,
                  'H_CYTO_INIT': 0,
                  'P_CYTO_INIT': 0,
                  'G_EXT_INIT': data_sample_df[0, 0],
                  'H_EXT_INIT': 0,
                  'P_EXT_INIT': data_sample_df[0, 1],
                  }  # set initial conditions

    # plot and get exp_data_native statistics
    _,_,median_sol = dhaB_dhaT_model.generate_time_series(DCW_time_data[gly_cond], DCW_data[gly_cond], init_conds,
                                                             median_param_dict, evaluation_times=teval)
    save_obj(median_sol, directory_data_formatted + '/median_param_solution_array_burn_in_' + str(burn_in_n)
                 + '_skip_' + str(skip) + '_ntimepts_' + str(ntimepts)+ '_init_cond_' + str(gly_cond))
    _,_,mean_sol = dhaB_dhaT_model.generate_time_series(DCW_time_data[gly_cond], DCW_data[gly_cond], init_conds,
                                                             mean_param_dict, evaluation_times=teval)
    save_obj(mean_sol, directory_data_formatted + '/mean_param_solution_array_burn_in_' + str(burn_in_n)
                 + '_skip_' + str(skip) + '_ntimepts_' + str(ntimepts)+ '_init_cond_' + str(gly_cond))
