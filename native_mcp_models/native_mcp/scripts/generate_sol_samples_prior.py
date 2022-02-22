import matplotlib as mpl
import numpy as np

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from mpi4py import MPI
from pdo_model_sympy import *
from pdo_model_sympy import TOTAL_PARAMETERS, NORM_PRIOR_MEAN_SINGLE_EXP,SINGLE_EXP_CALIBRATION_LIST
from exp_data import TIME_SAMPLES, DATA_SAMPLES, STD_EXPERIMENTAL_DATA
from scipy.stats import multivariate_normal
from os.path import dirname, abspath
import pickle

ROOT_PATH = dirname(abspath(__file__))

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    np.random.seed(1)
    #find maxtime
    maxtime = -np.inf
    for val in TIME_SAMPLES.values():
        maxtime = np.max([np.max(val),maxtime])
    maxtime = np.ceil(maxtime) + 10
    ntimepts = 250
    sol_times = np.linspace(0, maxtime, ntimepts)
    nsamples = int(1e5)
    samples = multivariate_normal.rvs(mean=np.zeros(N_MODEL_PARAMETERS + 4),
                                     cov=np.diag(np.ones(N_MODEL_PARAMETERS + 4)), size=nsamples)

    # generate param samples split
    nsamples_rank0 = nsamples // size + nsamples % size
    nsamples_rank = nsamples // size
    count_scatter = [nsamples_rank0]
    count_scatter.extend((size - 2) * [nsamples_rank])
    count_scatter = np.cumsum(count_scatter)

    samples_split = np.split(samples, count_scatter)
else:
    samples_split = None
    sol_times = None

samples_rank = comm.scatter(samples_split,root=0)
sol_times = comm.bcast(sol_times, root=0)
pdo_model = pdo_model_log()

sol_conc = []
mse_array = []
for params in samples_rank:
    sol_exp = []
    mse_exp = []
    params_ext = np.concatenate((params,np.zeros((N_DCW_PARAMETERS*4,))))
    for i, gly_cond in enumerate([50, 60, 70, 80]):
        evaluation_times = TIME_SAMPLES[gly_cond]
        # get parameters and rescale for exp
        exp_params_std = [*params_ext[:N_MODEL_PARAMETERS], params_ext[N_MODEL_PARAMETERS + i],
                          *params_ext[(N_MODEL_PARAMETERS + 4 + N_DCW_PARAMETERS * i):(N_MODEL_PARAMETERS + 4
                                                                                   + N_DCW_PARAMETERS * (i + 1))]]
        exp_params = np.matmul(NORM_PRIOR_STD_RT_SINGLE_EXP[gly_cond].T, exp_params_std) \
                     + NORM_PRIOR_MEAN_SINGLE_EXP[gly_cond]

        # store params as dictionary
        params_dict = {}
        for param, param_name in zip(exp_params,SINGLE_EXP_CALIBRATION_LIST):
            params_dict[param_name] = param
        init_conds = {'G_CYTO_INIT': 0., 'H_CYTO_INIT': 0., 'P_CYTO_INIT': 0.,'H_EXT_INIT': 0.,
                      'P_EXT_INIT': DATA_SAMPLES[gly_cond][0,1]}

        all_params_dict = {**params_dict,**init_conds}

        # arguments to get time series
        experimental_data_mat = DATA_SAMPLES[gly_cond][:,:2]
        sigma_exp_matrix = STD_EXPERIMENTAL_DATA[gly_cond][:,:2]
        tmatrix = NORM_PRIOR_STD_RT_SINGLE_EXP[gly_cond][:len(params),:len(params)]

        #get time series for given experiment
        FLAG, _, sols = pdo_model.get_sol_sens_time_series(all_params_dict, evaluation_times=sol_times, type='qoi only')
        qoi_mse, _ = pdo_model.get_loglik_single_exp(exp_params, evaluation_times, experimental_data_mat,
                                                            sigma_exp_matrix, tmatrix, type='qoi only')
        if FLAG == 0:
            sol_exp.append(sols)
            mse_exp.append(qoi_mse)
        else:
            sols = np.empty((sol_times.shape[0],6))
            sols[:] = np.nan
            sol_exp.append(sols)
            mse_exp.append(np.nan)

    sol_conc.append(np.array(sol_exp))
    mse_array.append(np.array(mse_exp))

sol_conc = np.array(sol_conc)
mse_array = np.array(mse_array)
sol_conc = comm.gather(sol_conc,root=0)
mse_array = comm.gather(mse_array,root=0)

if rank == 0:
    sol_conc = np.concatenate(sol_conc)
    mse_array = np.concatenate(mse_array)

    sol_folder_name_data = ROOT_PATH + '/data/sol_conc_prior_nsamples_' + str(nsamples) + '_ntimepts_' + str(ntimepts) + '_maxtime_' + str(int(maxtime))
    with open(sol_folder_name_data + '.pkl', 'wb') as f:
        pickle.dump(sol_conc, f, pickle.HIGHEST_PROTOCOL)

    mse_folder_name_data = ROOT_PATH + '/data/mse_array_prior_nsamples_' + str(nsamples) + '_ntimepts_' + str(ntimepts) + '_maxtime_' + str(int(maxtime))
    with open(mse_folder_name_data + '.pkl', 'wb') as f:
        pickle.dump(mse_array, f, pickle.HIGHEST_PROTOCOL)

    samples_folder_name_data = ROOT_PATH + '/data/samples_prior_nsamples_' + str(nsamples) + '_ntimepts_' + str(ntimepts) + '_maxtime_' + str(int(maxtime))
    with open(samples_folder_name_data + '.pkl', 'wb') as f:
        pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)