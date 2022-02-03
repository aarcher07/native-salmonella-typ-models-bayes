from pdo_model_pytorch.misc_functions import load_obj, save_obj
from os.path import dirname, abspath
import os
import numpy as np
from plot_funcs_param_distribution import plot_posterior_density, plot_corr, plot_corr_scatter,filtered_plot_posterior_density,\
    plot_prior_density,plot_corr_density
ROOT_PATH = dirname(dirname(dirname(dirname(abspath(__file__)))))

burn_in_n = int(2e3)
skip = 10

# create file names
directory_data_formatted = ROOT_PATH + "/MCMC/output/MCMC_results_formatted/adaptive/preset_std/lambda_0,01_beta_0,25/nsamples_10000/norm"
directory_plot = '/'.join([st if not st == "MCMC_results_formatted" else "MCMC_results_plots" for st in directory_data_formatted.split('/')]) + '/interpretation_plots'
data_dict = load_obj(directory_data_formatted + '/data_burnin_' + str(burn_in_n) + '_skip_' + str(int(skip)))
filtered_data_dict = load_obj(directory_data_formatted + '/filtered_data_burnin_' + str(burn_in_n) + '_skip_' + str(int(skip)))
#plot_data_density(data_dict,directory_plot,'wo_filter', bounds=False)
#filtered_plot_data_density(data_dict,directory_plot,'w_filter', bounds=False)
plot_prior_density(ROOT_PATH + "/MCMC/output/prior_plots")
#plot_corr(data_dict,directory_plot,'wo_filter')
#plot_corr_scatter(data_dict,directory_plot,'wo_filter')
#plot_corr_density(data_dict,directory_plot,'wo_filter')
