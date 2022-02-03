import os
from pdo_model_pytorch import MODEL_PARAMETER_LIST, NORM_PRIOR_MEAN_ALL_EXP, NORM_PRIOR_STD_RT_ALL_EXP, \
    VARS_ALL_EXP_TO_UNITS, VARS_ALL_EXP_TO_TEX
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from os.path import dirname, abspath
import pickle

ROOT_PATH = dirname(abspath(__file__))
directory_data = ROOT_PATH + '/output/MCMC_results_data/'
# get exp_data_native and create relevant files
for subdir, dirs, files in os.walk(directory_data):
    for file in files:
        if file.endswith('.pkl'):
            data_pkl_path = os.path.join(subdir, file)
            # create path if doesnt exist
            dir_plot_path = subdir.replace('output', 'plot')
            if not os.path.exists(dir_plot_path):
                os.makedirs(dir_plot_path)

            # check if directory is empty
            if not os.listdir(dir_plot_path):
                # load data
                with open(data_pkl_path, 'rb') as f:
                    postdraws = pickle.load(f)

                # transform data
                params_trans = []
                for param in postdraws["samples"]:
                    param_trans = np.matmul(NORM_PRIOR_STD_RT_ALL_EXP.T, param) + NORM_PRIOR_MEAN_ALL_EXP
                    params_trans.append(param_trans)
                params_trans = np.array(params_trans)

                for i,param_name in enumerate(VARS_ALL_EXP_TO_UNITS.keys()):
                    vars_units = VARS_ALL_EXP_TO_UNITS[param_name]
                    vars_tex = VARS_ALL_EXP_TO_TEX[param_name]
                    plt.plot(range(int(params_trans.shape[0])), params_trans[:, i])
                    plt.xlabel('iterations index')
                    if param_name in MODEL_PARAMETER_LIST:
                        plt.title(
                            'Plot of MCMC distribution of ' + r'$\log($' + vars_tex + '$)$')
                        plt.ylabel(r'$\log($' + vars_tex + '$)$')
                    else:
                        plt.title('Plot of MCMC distribution of ' + VARS_ALL_EXP_TO_TEX[param_name])
                        plt.ylabel(VARS_ALL_EXP_TO_TEX[param_name] + " (" + vars_units + ")")

                    plot_path = os.path.join(dir_plot_path, file[:-4]) + "_param_" + param_name + '.png'
                    plt.savefig(plot_path, bbox_inches='tight')
                    plt.close()
