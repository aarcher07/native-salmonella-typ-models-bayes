import os
from pdo_model_pytorch.misc_functions import load_obj
from pdo_model_pytorch.model_constants import MODEL_PARAMETER_LIST,VARS_TO_TEX
from pdo_model_pytorch.data_set_constants import INIT_CONDS_GLY_PDO_DCW,INTERSECTION_TIME_EVALS,DATA_SAMPLES,TIME_SAMPLES
from MCMC.dhaB_dhaT_model_prior import DhaBDhaTModelMCMC
from MCMC.dhaB_dhaT_model_bayesian_funs import loglik
from gaussian_process_design import LOG_PARAMETERS_BOUNDS

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
import numpy as np

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

directory = "/home/aarcher/Dropbox/PycharmProjects/13_PDO_Pathway_Inference/MCMC/output/MCMC_results_emulator_data/complete_transform_log_unif_nsamples_250_tol_1e-05_date_2021_07_26_17:38/adaptive/preset_std/lambda_0,1_beta_0,01/nsamples_100000/"#"/home/aarcher/Dropbox/PycharmProjects/13_PDO_Pathway_Inference/MCMC/output/MCMC_results_data/adaptive/preset_std/lambda_0,1_beta_0,01/nsamples_10000/norm/"
burn_in_n = int(1e4)
data_dict = {parameter:[] for parameter in MODEL_PARAMETER_LIST if parameter != 'DCW'}
for filename in os.listdir(directory):
    data = load_obj(directory + "/" +filename[:-4])
    for i in range(data.shape[1]):
        data_dict[MODEL_PARAMETER_LIST[i]].extend((LOG_PARAMETERS_BOUNDS[MODEL_PARAMETER_LIST[i]][1]-LOG_PARAMETERS_BOUNDS[MODEL_PARAMETER_LIST[i]][0])*data[range(burn_in_n,int(1e5),100),i]+LOG_PARAMETERS_BOUNDS[MODEL_PARAMETER_LIST[i]][0])

#save_file_name = "adaptive/sigma_[2,2,0,2]/lambda_0,01_beta_0,01/nsamples_100000/norm/"
mean_params = []

for parameter in MODEL_PARAMETER_LIST:
    if parameter != 'DCW':
        sns.histplot(data=data_dict[parameter],stat='probability', bins='auto', color='#0504aa',alpha=0.7)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('$\log_{10}($'+VARS_TO_TEX[parameter]+'$)$')
        plt.title('Histogram of MCMC walk of $\log_{10}($'+ VARS_TO_TEX[parameter]+'$)$')
        plt.axvline(x=np.mean(data_dict[parameter]), color='red',linewidth=4)
        plt.axvline(x=LOG_PARAMETERS_BOUNDS[parameter][0], color='orange',linewidth=4)
        plt.axvline(x=LOG_PARAMETERS_BOUNDS[parameter][1], color='orange',linewidth=4)
        mean_params.append(np.mean(data_dict[parameter]))
        plt.show()

dhaB_dhaT_model = DhaBDhaTModelMCMC(transform="log_norm")
mean_param_dict = {param_name:param for param_name,param in zip(MODEL_PARAMETER_LIST,mean_params) if param_name != 'DCW'}
DCW_time_data, DCW_data = load_obj(
    '/home/aarcher/Dropbox/PycharmProjects/13_PDO_Pathway_Inference/exp_data_native/data_files/DCW_discretized_N_100')

for gly_cond in TIME_SAMPLES.keys():
    data_sample_df = DATA_SAMPLES[gly_cond]  # experimental exp_data_native
    tsamp = TIME_SAMPLES[gly_cond]

    init_conds = {'G_CYTO_INIT': 0,
                  'H_CYTO_INIT': 0,
                  'P_CYTO_INIT': 0,
                  'G_EXT_INIT': data_sample_df[0, 0],
                  'H_EXT_INIT': 0,
                  'P_EXT_INIT': data_sample_df[0, 1],
                  }  # set initial conditions

    qoi_vals = dhaB_dhaT_model.QoI(DCW_time_data[gly_cond], DCW_data[gly_cond], init_conds, mean_param_dict, tsamp)
    plt.plot(tsamp, qoi_vals[:, 0])
    plt.scatter(tsamp,data_sample_df[:, 0])
    plt.title('Plot of external glycerol')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    plt.plot(tsamp, qoi_vals[:, 1])
    plt.scatter(tsamp,DATA_SAMPLES[gly_cond][:,1])
    plt.title('Plot of external 1,3-PDO')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()
