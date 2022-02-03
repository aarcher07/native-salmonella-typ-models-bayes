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

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    ROOT_PATH = dirname(dirname(dirname(dirname(abspath(__file__)))))
    burn_in_n = int(2e3)
    skip = 10
    # create filenames and load files names
    directory_data_formatted = ROOT_PATH + "/MCMC/output/MCMC_results_formatted/adaptive/preset_std/lambda_0,01_beta_0,25/nsamples_10000/norm"
    directory_plot = '/' + '/'.join([st if not st == "MCMC_results_formatted" else "MCMC_results_plots" for st in directory_data_formatted.split('/')]) + '/interpretation_plots'
    data_dict = load_obj(directory_data_formatted + '/data_burnin_' + str(burn_in_n) + '_skip_' + str(int(skip)))
    sigma = np.loadtxt(ROOT_PATH +
                       "/exp_data_native/data_files/estimated_std_milliMolar_data.csv", delimiter=",")
    # reformat to array
    parameter_samples_array = np.array([val for val in data_dict.values()]).T
    median_parameter_samples_array = np.median(parameter_samples_array, axis=0)

    nsamples = parameter_samples_array.shape[0]
    niters_rank0 = nsamples // size + nsamples % size
    niters_rank = nsamples // size

    # load DCW exp_data_native
    DCW_time_data, DCW_data = load_obj(ROOT_PATH + '/exp_data_native/data_files/DCW_discretized_N_100')

    # parameters for scatter
    if size > 1:
        count_scatter = [niters_rank0]
        count_scatter.extend((size - 2) * [niters_rank])
        count_scatter = np.cumsum(count_scatter)
        param_samples_split = np.split(parameter_samples_array, count_scatter)
else:
    param_samples_split = None
    DCW_time_data = None
    DCW_data = None
    sigma = None

if size > 1:
    param_samples_rank = comm.scatter(param_samples_split, root=0)
    DCW_time_data = comm.bcast(DCW_time_data, root=0)
    DCW_data = comm.bcast(DCW_data, root=0)
    sigma = comm.bcast(sigma,root=0)
else:
    param_samples_rank = parameter_samples_array

# create model
dhaB_dhaT_model = DhaBDhaTModelMCMC(transform="log_norm")
tol=10**-8
all_exp_lik = np.zeros(len(param_samples_rank))
ntotprev_entries = 0
for gly_cond in DATA_SAMPLES.keys():
    sol_array = []
    data_sample_df = DATA_SAMPLES[gly_cond]  # experimental exp_data_native
    tsamp = TIME_SAMPLES[gly_cond]
    exp_lik = []
    nentries = data_sample_df.shape[0]*data_sample_df.shape[1]
    sigma_subset = sigma[ntotprev_entries:(ntotprev_entries + nentries)]
    ntotprev_entries += nentries

    for i,param_sample in enumerate(param_samples_rank):
        param_dict = {param_name:param for param_name,param in zip(SINGLE_EXP_CALIBRATION_LIST, param_sample)}
        init_conds = {'G_CYTO_INIT': 0,
                      'H_CYTO_INIT': 0,
                      'P_CYTO_INIT': 0,
                      'G_EXT_INIT': data_sample_df[0, 0],
                      'H_EXT_INIT': 0,
                      'P_EXT_INIT': data_sample_df[0, 1],
                      }  # set initial conditions

        FLAG,  fvals = dhaB_dhaT_model.QoI(DCW_time_data[gly_cond], DCW_data[gly_cond], init_conds,
                                           param_dict, tsamp)
        if FLAG == 0:
            data_diff_vec = (fvals - data_sample_df).ravel()
            loglik_param = -0.5*np.dot(data_diff_vec,data_diff_vec)/sigma_subset
            all_exp_lik[i] += loglik_param
            exp_lik.append(loglik_param)
        else:
            all_exp_lik[i] += -np.inf
            exp_lik.append(-np.inf)
    exp_lik_array = comm.gather(exp_lik, root=0)

    if rank == 0:
        # save output
        exp_lik_array_concat = np.concatenate(exp_lik_array)
        save_obj(exp_lik_array_concat, directory_data_formatted + '/exp_lik_array_burn_in_' + str(burn_in_n)
                 + '_skip_' + str(skip) + '_init_cond_' + str(gly_cond))

all_exp_lik_array = comm.gather(all_exp_lik, root=0)

if rank == 0:
    all_exp_lik_array_concat = np.concatenate(all_exp_lik_array)
    print(all_exp_lik_array_concat)
    save_obj(all_exp_lik_array_concat, directory_data_formatted + '/exp_lik_array_burn_in_' + str(burn_in_n)
             + '_skip_' + str(skip))
