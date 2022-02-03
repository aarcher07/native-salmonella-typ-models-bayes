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
import matplotlib.colors as mcolors
import signal
from sklearn.manifold import TSNE,LocallyLinearEmbedding,SpectralEmbedding
from pdo_model_pytorch.data_set_constants import DATA_SAMPLES, N_REACTANT_DATA_POINTS
from sklearn.preprocessing import StandardScaler

ROOT_PATH = dirname(dirname(dirname(dirname(abspath(__file__)))))
burn_in_n = int(2e3)
skip = 10
directory_data_formatted = ROOT_PATH + "/MCMC/output/MCMC_results_formatted/adaptive/preset_std/lambda_0,01_beta_0,25/nsamples_10000/norm"
directory_plot = '/' + '/'.join([st if not st == "MCMC_results_formatted" else "MCMC_results_plots" for st in directory_data_formatted.split('/')]) + '/interpretation_plots'
data_dict = load_obj(directory_data_formatted + '/data_burnin_' + str(burn_in_n) + '_skip_' + str(int(skip)))
# reformat to array
parameter_samples_array = np.array([val for val in data_dict.values()]).T
scaler = StandardScaler()
subset_bool = np.array([True,True,False,True,False,True,False,True,False])
standard_parameter_samples_array = scaler.fit_transform(parameter_samples_array[:,subset_bool])
X_embedded = TSNE(n_components=2).fit_transform(standard_parameter_samples_array)
all_exp_lik = np.zeros(parameter_samples_array.shape[0])

for gly_cond in DATA_SAMPLES.keys():
    exp_lik_array = np.array(load_obj(directory_data_formatted + '/exp_lik_array_burn_in_' + str(burn_in_n) + '_skip_' + str(skip) + '_init_cond_' + str(gly_cond)))
    if gly_cond < 80:
        threshold1 = -10
        bool = exp_lik_array > threshold1
        threshold = threshold1
    else:
        threshold2 = -40
        bool = exp_lik_array > threshold2
        threshold = threshold2
    plt.scatter(X_embedded[bool,0],X_embedded[bool,1], c = exp_lik_array[bool], cmap = 'seismic')
    plt.colorbar()
    plt.title('Embedding of Posterior Samples colored by log-likelihood \n with initial glycerol ' + str(gly_cond) + ' g/L')
    plt.savefig(directory_plot + '/embedding_posterior_samples_burnin_'+str(burn_in_n) + '_skip_' + str(skip) +
                '_thres_' + str(int(threshold)) + '_' + str(gly_cond), bbox_inches="tight")
    plt.close()
all_exp_lik = np.array(load_obj(directory_data_formatted + '/exp_lik_array_burn_in_' + str(burn_in_n)+ '_skip_' + str(skip)))
threshold = threshold2 + threshold1*3
bool = all_exp_lik > threshold
plt.scatter(X_embedded[bool,0],X_embedded[bool,1], c = all_exp_lik[bool], cmap = 'seismic')
plt.colorbar()
plt.title('Embedding of Posterior Samples colored by log-likelihood\n across all experiments')
plt.savefig(directory_plot + '/embedding_posterior_samples_burnin_' + str(burn_in_n) + '_skip_' + str(skip) +
            '_thres_' + str(int(threshold)), bbox_inches="tight")
plt.close()
