import os
from pdo_model_pytorch.misc_functions import load_obj
from pdo_model_pytorch.model_constants import MODEL_PARAMETER_LIST,VARS_TO_TEX,SINGLE_EXP_CALIBRATION_LIST
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
from pandas.plotting import scatter_matrix
import corner

def plot_posterior_density(data_dict, directory_plot, plot_preamble, bounds = True):
    for parameter in MODEL_PARAMETER_LIST:
        if parameter != 'DCW':
            fig, ax = plt.subplots()
            #prior density plot
            vals = LOG_NORM_PRIOR_PARAMETERS[parameter]
            prior_samples = stats.norm.rvs(size=int(1e6),loc=vals[0],scale=vals[1])
            sns.kdeplot(prior_samples, fill=True, alpha=0.5, color='red',   label='Prior Density', ax=ax)

            #posterior histogram plot
            sns.kdeplot(data=data_dict[parameter], fill=True,
                         color='blue',alpha=0.5, label="Posterior Density",ax=ax)
            plt.grid(axis='y', alpha=0.75)
            plt.axvline(x=np.median(data_dict[parameter]), color='blue',linewidth=4, label="Posterior median")
            plt.axvline(x=np.median(prior_samples), color='red',linewidth=4, label="Prior median")
            ylim = ax.get_ylim()
            if bounds:
                y = np.linspace(ylim[0],ylim[1]+0.01*ylim[1])
                if np.isnan(DATA_LOG_UNIF_PARAMETER_RANGES[parameter][0]):
                    lower_x = ax.get_xlim()[0]
                else:
                    lower_x = DATA_LOG_UNIF_PARAMETER_RANGES[parameter][0]
                if np.isnan(DATA_LOG_UNIF_PARAMETER_RANGES[parameter][1]):
                    upper_x = ax.get_xlim()[1]
                else:
                    upper_x = DATA_LOG_UNIF_PARAMETER_RANGES[parameter][1]
                ax.fill_betweenx(y,lower_x,upper_x,color='green',
                                 alpha=0.5, label="Range from literature")

            #plot parameters
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.legend(fontsize= 10)
            plt.xlabel('$\log_{10}($'+VARS_TO_TEX[parameter]+'$)$', fontsize=20)
            plt.ylabel('Probability', fontsize=20)
            plt.title('Distribution of $\log_{10}($'+ VARS_TO_TEX[parameter]+'$)$',fontsize= 20)
            plt.savefig(directory_plot + '/data_density_' + parameter + "_w_prior_" + plot_preamble, bbox_inches="tight")
            plt.close()

            fig, ax = plt.subplots()
            #posterior histogram plot
            sns.kdeplot(data=data_dict[parameter], fill=True,
                        color='blue',alpha=0.5, label="Posterior Density",ax=ax)
            plt.grid(axis='y', alpha=0.75)
            plt.axvline(x=np.median(data_dict[parameter]), color='blue',linewidth=4, label="Posterior median")
            ylim = ax.get_ylim()
            y = np.linspace(ylim[0],ylim[1]+0.01*ylim[1])
            if np.isnan(DATA_LOG_UNIF_PARAMETER_RANGES[parameter][0]):
                lower_x = ax.get_xlim()[0]
            else:
                lower_x = DATA_LOG_UNIF_PARAMETER_RANGES[parameter][0]
            if np.isnan(DATA_LOG_UNIF_PARAMETER_RANGES[parameter][1]):
                upper_x = ax.get_xlim()[1]
            else:
                upper_x = DATA_LOG_UNIF_PARAMETER_RANGES[parameter][1]
            ax.fill_betweenx(y,lower_x,upper_x,color='green',
                             alpha=0.5, label="Range from literature")
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.legend(fontsize= 10)
            plt.xlabel('$\log_{10}($'+VARS_TO_TEX[parameter]+'$)$', fontsize=20)
            plt.ylabel('Probability', fontsize=20)
            plt.title('Distribution of $\log_{10}($'+ VARS_TO_TEX[parameter]+'$)$',fontsize= 20)
            plt.savefig(directory_plot + '/data_density_' + parameter + "_w_data_bounds_" + plot_preamble, bbox_inches="tight")
            plt.close()

def plot_prior_density(directory_plot):
    for parameter in MODEL_PARAMETER_LIST:
        if parameter != 'DCW':
            fig, ax = plt.subplots()
            #prior density plot
            vals = LOG_NORM_PRIOR_PARAMETERS[parameter]
            prior_samples = stats.norm.rvs(size=int(1e6),loc=vals[0],scale=vals[1])
            sns.kdeplot(prior_samples, fill=True, alpha=0.5, color='red',   label='Prior Density', ax=ax)
            plt.axvline(x=np.median(prior_samples), color='red',linewidth=4,
                        label="Prior median")
            #plot parameters
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.legend(fontsize= 10)
            plt.xlabel('$\log_{10}($'+VARS_TO_TEX[parameter]+'$)$', fontsize=20)
            plt.ylabel('Probability', fontsize=20)
            plt.title('Distribution of $\log_{10}($'+ VARS_TO_TEX[parameter]+'$)$',fontsize= 20)
            plt.savefig(directory_plot + '/prior_density_' + parameter, bbox_inches="tight")
            plt.close()

            fig, ax = plt.subplots()
            sns.kdeplot(prior_samples, fill=True, alpha=0.5, color='red',
                        label='Prior Density', ax=ax)
            plt.axvline(x=np.median(prior_samples), color='red',linewidth=4,
                        label="Prior median")
            ylim = ax.get_ylim()
            y = np.linspace(ylim[0],ylim[1]+0.01*ylim[1])
            if np.isnan(DATA_LOG_UNIF_PARAMETER_RANGES[parameter][0]):
                lower_x = ax.get_xlim()[0]
            else:
                lower_x = DATA_LOG_UNIF_PARAMETER_RANGES[parameter][0]
            if np.isnan(DATA_LOG_UNIF_PARAMETER_RANGES[parameter][1]):
                upper_x = ax.get_xlim()[1]
            else:
                upper_x = DATA_LOG_UNIF_PARAMETER_RANGES[parameter][1]
            ax.fill_betweenx(y,lower_x,upper_x,color='blue',
                             alpha=0.5, label="Range from literature")
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.legend(fontsize= 10)
            plt.xlabel('$\log_{10}($'+VARS_TO_TEX[parameter]+'$)$', fontsize=20)
            plt.ylabel('Probability', fontsize=20)
            plt.title('Distribution of $\log_{10}($'+ VARS_TO_TEX[parameter]+'$)$',fontsize= 20)
            plt.savefig(directory_plot + '/prior_density_' + parameter + "_w_literature_bounds", bbox_inches="tight")
            plt.close()

def filtered_plot_posterior_density(data_dict, directory_plot, plot_preamble, bounds = True):
    bools = np.ones(len(list(data_dict.values())[0])).astype('int')

    for parameter in ['PermCellGlycerol','PermCellPDO']:
       param_data = data_dict[parameter]
       bools = bools*(DATA_LOG_UNIF_PARAMETER_RANGES[parameter][0] < param_data)*(param_data < DATA_LOG_UNIF_PARAMETER_RANGES[parameter][1])
    bools = bools*(data_dict["PermCell3HPA"] < DATA_LOG_UNIF_PARAMETER_RANGES["PermCell3HPA"][1])

    for parameter in MODEL_PARAMETER_LIST:
        if parameter != 'DCW':
            fig, ax = plt.subplots()
            #prior density plot
            filtered_data = [param for (param, bool) in zip(data_dict[parameter], bools) if bool]
            sns.kdeplot(filtered_data, fill=True, alpha=0.5, color='red',   label='Filtered Posterior Density', ax=ax)

            #posterior histogram plot
            sns.kdeplot(data=data_dict[parameter], fill=True,
                        color='blue',alpha=0.5, label="Posterior Density",ax=ax)
            plt.grid(axis='y', alpha=0.75)
            plt.axvline(x=np.median(data_dict[parameter]), color='blue',linewidth=4, label="Posterior median")
            plt.axvline(x=np.median(filtered_data), color='red',linewidth=4, label="Filtered posterior median")
            ylim = ax.get_ylim()
            _, pvalue = stats.ks_2samp(filtered_data, data_dict[parameter])
            print(pvalue)
            if bounds:
                y = np.linspace(ylim[0],ylim[1]+0.01*ylim[1])
                if np.isnan(DATA_LOG_UNIF_PARAMETER_RANGES[parameter][0]):
                    lower_x = ax.get_xlim()[0]
                else:
                    lower_x = DATA_LOG_UNIF_PARAMETER_RANGES[parameter][0]
                if np.isnan(DATA_LOG_UNIF_PARAMETER_RANGES[parameter][1]):
                    upper_x = ax.get_xlim()[1]
                else:
                    upper_x = DATA_LOG_UNIF_PARAMETER_RANGES[parameter][1]
                ax.fill_betweenx(y,lower_x,upper_x,color='green',
                                 alpha=0.5, label="Range from literature")

            if not parameter in ['PermCellGlycerol','PermCellPDO', 'PermCell3HPA']:
                plt.annotate(r'$p =' + "{0:.2g}".format(pvalue) + '$', xy=(0.77,0.7),
                             xycoords='axes fraction',fontsize=22)
            #plot parameters
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.legend(fontsize= 10,loc='upper left')
            plt.xlabel('$\log_{10}($'+VARS_TO_TEX[parameter]+'$)$', fontsize=20)
            plt.ylabel('Probability', fontsize=20)
            plt.title('Distribution of $\log_{10}($'+ VARS_TO_TEX[parameter]+'$)$',fontsize= 20)
            plt.savefig(directory_plot + '/filtered_posterior_density_' + parameter + "_w_prior_" + plot_preamble, bbox_inches="tight")
            plt.close()

            fig, ax = plt.subplots()
            #posterior histogram plot
            sns.kdeplot(data=filtered_data, fill=True,
                        color='blue',alpha=0.5, label="Filtered Posterior Density",ax=ax)
            plt.grid(axis='y', alpha=0.75)
            plt.axvline(x=np.median(filtered_data), color='blue',linewidth=4, label="Filtered posterior median")
            ylim = ax.get_ylim()
            y = np.linspace(ylim[0],ylim[1]+0.01*ylim[1])

            if np.isnan(DATA_LOG_UNIF_PARAMETER_RANGES[parameter][0]):
                lower_x = ax.get_xlim()[0]
            else:
                lower_x = DATA_LOG_UNIF_PARAMETER_RANGES[parameter][0]
            if np.isnan(DATA_LOG_UNIF_PARAMETER_RANGES[parameter][1]):
                upper_x = ax.get_xlim()[1]
            else:
                upper_x = DATA_LOG_UNIF_PARAMETER_RANGES[parameter][1]
            ax.fill_betweenx(y,lower_x,upper_x,color='green',
                             alpha=0.5, label="Range from literature")
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.legend(fontsize= 10)
            plt.xlabel('$\log_{10}($'+VARS_TO_TEX[parameter]+'$)$', fontsize=20)
            plt.ylabel('Probability', fontsize=20)
            plt.title('Distribution of $\log_{10}($'+ VARS_TO_TEX[parameter]+'$)$',fontsize= 20)
            plt.savefig(directory_plot + '/filtered_posterior_density_' + parameter + "_w_literature_bounds_" + plot_preamble, bbox_inches="tight")
            plt.close()

def plot_corr(data_dict, directory_plot, plot_preamble):
    fig, ax = plt.subplots()
    data_values=np.array(list(data_dict.values()))
    data_corr = np.corrcoef(data_values)
    matrix = np.triu(data_corr)
    # using the upper triangle matrix as mask
    ax = sns.heatmap(data_corr, mask=matrix, annot=True, cmap="YlGnBu", vmin=-1, vmax=1, annot_kws={"size":20},fmt='.1g')
    xticks = [(i + 0.5) for i in range(len(SINGLE_EXP_CALIBRATION_LIST) - 1)]
    yticks = [(i + 0.5) for i in range(1, len(SINGLE_EXP_CALIBRATION_LIST))]
    plt.xticks(xticks, [VARS_TO_TEX[key] for i,key in enumerate(data_dict.keys()) if not i == len(data_dict.keys())-1], fontsize=20,
               rotation = -25)
    plt.yticks(yticks, [VARS_TO_TEX[key] for i,key in enumerate(data_dict.keys()) if not i == 0], fontsize=20,
               rotation = 45, ha="right")
    plt.title('Correlation Matrix of Posterior Samples', fontsize=20)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    fig.set_size_inches(9.5, 9.5, forward=True)
    plt.savefig(directory_plot + '/correlation_plot_' + plot_preamble, bbox_inches="tight")
    plt.close()

def plot_corr_scatter(data_dict, directory_plot, plot_preamble):
    data_array = pd.DataFrame(np.array([val for key, val in data_dict.items()]).T,columns=list(VARS_TO_TEX.values()))
    axes = scatter_matrix(data_array, alpha = 0.2, figsize = (len(SINGLE_EXP_CALIBRATION_LIST), len(SINGLE_EXP_CALIBRATION_LIST)), diagonal ='kde')
    for i in range(np.shape(axes)[0]):
        for j in range(np.shape(axes)[1]):
            if i < j:
                axes[i,j].set_visible(False)
            axes[i,j].xaxis.label.set_rotation(-25)
            axes[i,j].xaxis.label.set_fontsize(20)
            axes[i,j].yaxis.label.set_rotation(45)
            axes[i,j].yaxis.label.set_ha('right')
            axes[i,j].yaxis.label.set_fontsize(20)
            axes[i,j].tick_params(axis="x",labelsize=10,rotation=0)
            axes[i,j].tick_params(axis="y",labelsize=10)

    plt.suptitle('Scatter Plot Matrix of Posterior Samples', fontsize=20)
    plt.savefig(directory_plot + '/correlation_scatter_plot_' + plot_preamble, bbox_inches="tight")
    plt.close()

def plot_corr_density(data_dict, directory_plot, plot_preamble):
    data_array = pd.DataFrame(np.array([val for key, val in data_dict.items()]).T,columns=list(VARS_TO_TEX.values()))
    figure = corner.corner(data_array, labels=[VARS_TO_TEX[key] for key in data_dict.keys()],
                           quantiles=[0.05, 0.5, 0.95],
                           show_titles=True, title_kwargs={"fontsize": 12},smooth=1)
    axes = figure.axes
    for i in range(len(axes)):
        axes[i].yaxis.label.set_fontsize(35)
        axes[i].xaxis.label.set_fontsize(35)
        axes[i].title.set_fontsize(18)
        axes[i].tick_params(axis="x",labelsize=18)
        axes[i].tick_params(axis="y",labelsize=18,rotation=0)
    plt.suptitle('Corner Plot of Posterior Samples', fontsize=50)
    plt.savefig(directory_plot + '/corner_plot_' + plot_preamble, bbox_inches="tight")
    plt.close()

