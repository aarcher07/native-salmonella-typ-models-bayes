from pdo_model_sympy import pdo_model_log, NORM_PRIOR_MEAN_SINGLE_EXP, N_MODEL_PARAMETERS, N_DCW_PARAMETERS,\
    NORM_PRIOR_STD_RT_SINGLE_EXP, N_UNKNOWN_PARAMETERS
from exp_data import INIT_CONDS_GLY_PDO_DCW,TIME_SAMPLES,DATA_SAMPLES
import time
from multiprocessing import Pool
from exp_data_native import DATA_SAMPLES, STD_EXPERIMENTAL_DATA, TIME_SAMPLES_BOBIK
import numpy as np

mcp_model = pdo_model_log()

def qoi_mse_grad(exp_cond, params, atol, rtol, type):
    """
    Gets loglik and derivative of loglik for single experimental

    @param exp_ind: 'Broken MCPs or WT'
    @param params: list of standardized parameters to evaluate differential equation
    @param atol: absolute tolerance
    @param rtol: relative tolerance
    @param type: qoi only or qoi sens
    @return loglik_all_exp: log-likelihood of single experiment
    @return dloglikdparam: derivative of log-likelihood of single experiment wrt
     params (N_UNKNOWN_PARAMETERS x 1  matrix)
    @return dloglikdparamdparam: Hessian of log-likelihood of single experiment wrt
     params (N_UNKNOWN_PARAMETERS x N_UNKNOWN_PARAMETERS  matrix)
    """

    evaluation_times = TIME_SAMPLES_BOBIK
    # get parameters and rescale for exp
    exp_params_std = params
    exp_params = np.matmul(NORM_PRIOR_STD_RT_SINGLE_EXP[exp_cond].T, exp_params_std) \
                 + NORM_PRIOR_MEAN_SINGLE_EXP[exp_cond]

    experimental_data_mat = DATA_SAMPLES[exp_cond]
    sigma_exp_matrix = STD_EXPERIMENTAL_DATA[exp_cond]
    tmatrix = NORM_PRIOR_STD_RT_SINGLE_EXP[exp_cond]

    # get loglik_all_exp
    qoi_mse, dloglikdparam, dloglikdparamdparam = mcp_model.get_loglik_single_exp(exp_params, evaluation_times,
                                                                                  experimental_data_mat,
                                                                                  sigma_exp_matrix, tmatrix, atol, rtol,
                                                                                  type)
    return qoi_mse, dloglikdparam, dloglikdparamdparam


def logpost(params, rtol=10 ** -3, atol=10 ** -6, type='qoi sens'):
    """
    Computes loglik and derivative of loglik

    @param params: list of standardized parameters to evaluate differential equation
    @param atol: absolute tolerance
    @param rtol: relative tolerance
    @param type: qoi only or qoi sens
    @return loglik_all_exp: log-likelihood of all experiment
    @return dloglikdparam_all_exp: derivative of log-likelihood of all experiment wrt
     params (N_UNKNOWN_PARAMETERS x 1  matrix)
    @return dloglikdparamdparam_all_exp: derivative of log-likelihood of all experiment wrt
     params (N_UNKNOWN_PARAMETERS x N_UNKNOWN_PARAMETERS matrix)
    """
    # TODO check transformation matrices
    # initialize
    loglik_all_exp, dloglikdparam_all_exp, dloglikdparamdparam_all_exp = qoi_mse_grad("WT", params, atol, rtol, type)
    # loglik_all_exp = 0
    # dloglikdparam_all_exp = np.zeros(N_UNKNOWN_PARAMETERS)
    # dloglikdparamdparam_all_exp = np.zeros((N_UNKNOWN_PARAMETERS, N_UNKNOWN_PARAMETERS))
    # with Pool(processes = 2) as pool:
    #     result_pool = pool.starmap(qoi_mse_grad, [(exp_ind, params, atol, rtol, type) for exp_ind in ['Broken MCPs', 'WT']])
    # for i,(qoi_mse, dloglikdparam, dloglikdparamdparam) in enumerate(result_pool):
    #     #sum mse values
    #     loglik_all_exp += qoi_mse
    #     if type == 'qoi sens':
    #         #store derivative values
    #         dloglikdparam_all_exp[:N_MODEL_PARAMETERS] += dloglikdparam[:N_MODEL_PARAMETERS]
    #         dloglikdparam_all_exp[N_MODEL_PARAMETERS + i] += dloglikdparam[N_MODEL_PARAMETERS]
    #
    #         #store hessian values
    #         dloglikdparamdparam_all_exp[:N_MODEL_PARAMETERS, :N_MODEL_PARAMETERS] += dloglikdparamdparam[:N_MODEL_PARAMETERS, :N_MODEL_PARAMETERS]
    #         dloglikdparamdparam_all_exp[N_MODEL_PARAMETERS + i, :N_MODEL_PARAMETERS] += dloglikdparamdparam[-1,:N_MODEL_PARAMETERS]
    #         dloglikdparamdparam_all_exp[:N_MODEL_PARAMETERS, N_MODEL_PARAMETERS + i] += dloglikdparamdparam[:N_MODEL_PARAMETERS, -1]
    #         dloglikdparamdparam_all_exp[N_MODEL_PARAMETERS + i, N_MODEL_PARAMETERS + i] += dloglikdparamdparam[-1, -1]
    # #prior loglik
    # loglik_all_exp -= np.dot(params[:(N_UNKNOWN_PARAMETERS)], params[:(N_UNKNOWN_PARAMETERS)]) /2.
    # dloglikdparam_all_exp -= params[:(N_UNKNOWN_PARAMETERS)]
    # dloglikdparamdparam_all_exp -= np.eye(N_UNKNOWN_PARAMETERS)

    if type == 'qoi only':
        return loglik_all_exp, None, None
    elif type == 'qoi sens':
        return loglik_all_exp, dloglikdparam_all_exp, dloglikdparamdparam_all_exp