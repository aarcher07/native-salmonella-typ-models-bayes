'''
The DhaB-DhaT model contains DhaB-DhaT reaction pathway
in the MCP; diffusion in the cell; diffusion from the cell 
in the external volume.

This model is currently in use. The DhaB-DhaT model assumes that there 
are N(t) identical cells within the external volume.

Programme written by aarcher07
Editing History:
- 28/10/20
'''


import sys

import numpy as np
import sympy as sp
from .model_constants import *
from scipy.integrate import solve_ivp
import signal
import matplotlib.pyplot as plt
from de_builder_sympy import LSAModelBuilder
from .native_mcp import pdo_model
from exp_data import TIME_SAMPLES, DATA_SAMPLES, STD_EXPERIMENTAL_DATA, INIT_CONDS_GLY_PDO_DCW
from .prior_constants import NORM_PRIOR_STD_RT_SINGLE_EXP, NORM_PRIOR_MEAN_SINGLE_EXP
import multiprocessing as mp

class TimeOutException(Exception):
    pass

def handler(signum, frame):
    raise TimeOutException("Integration is taking too long")

signal.signal(signal.SIGALRM, handler)

class pdo_model_log(pdo_model):

    def __init__(self, cell_volume = CELL_VOLUME, cell_surface_area = CELL_SURFACE_AREA,
                 external_volume = EXTERNAL_VOLUME):
        """
        Initializes parameters to be used numerial scheme
        :param cell_volume: cell volume in m^3
        :param cell_surface_area: cell surface area in m^2
        :param external_volume: external volume containing cells in metres^3
        """
        # geometric parameters
        super().__init__(cell_volume, cell_surface_area, external_volume)

        #integration parameters
        self.nvars = 2*3
        self.pdo_model = LSAModelBuilder(self._RHS, self.nvars, self.nparams, list(range(N_MODEL_PARAMETERS + 1)))

    def _RHS(self, t, x, log_params):
        """
        Computes the spatial derivative of the system at time point, t
        :param t: time
        :param x: state variables
        :param params: unstandardized parameters
        """
        params = []
        for param_name, param_val in zip(SINGLE_EXP_CALIBRATION_LIST, log_params):
            if param_name in MODEL_PARAMETER_LIST:
                params.append(10**param_val)
            else:
                params.append(param_val)

        d = super()._RHS(t, x, params)
        return d

    def get_sol_sens_time_series(self, params, evaluation_times = None, atol = 10**-6, rtol = 10**-3,
                                 type = 'qoi sens'):
        """
        Integrates the DhaB-DhaT model and sensitivity equation with parameter values, param, and returns external glycerol
        and 1,3-PDO time samples, tsamples

        @param params: dictionary of unstandardized parameters to evaluate differential equation
        @param evaluation_times: time samples to collect external glycerol, external 1,3-PDO and DCW
        @param atol: absolute tolerance
        @param rtol: relative tolerance

        @return FLAG: 0 if no issue with solving differential equation
                      1 if there was a ValueError
                      2 if there was a TimeException
                      3 if integration was unsuccessful
                      4 if mass growth
        @return t_concat: times at which the differential equations were solved
        @return sol_conat: differential equations solutions sampled at time samples, evaluation_times (3 x |evaluation_times| matrix)
        """
        # format inputs
        calibration_params = []
        for param_name in SINGLE_EXP_CALIBRATION_LIST:
            calibration_params.append(params[param_name])

        # ODE solver args initialization
        if type == 'qoi only':
            ds = lambda t,x: self.pdo_model.RHS(t, x, calibration_params)
            ds_jac = lambda t,x: self.pdo_model.RHS_jac_state_fun(t, x, calibration_params)

            #initial conditions
            y0 = np.zeros(N_VARIABLES)
            for i, init_names in enumerate(VARIABLE_INIT_NAMES):
                y0[i] = params[init_names]

        elif type == 'qoi sens':
            ds = lambda t,x: self.pdo_model.dsens(t, x, calibration_params)
            ds_jac = lambda t,x: self.pdo_model.dsens_jac(t, x, calibration_params)

            #initial conditions
            n_sens = self.nvars + self.nvars * self.pdo_model.nparams_sens
            y0 = np.zeros(n_sens)
            for i, init_names in enumerate(VARIABLE_INIT_NAMES):
                y0[i] = params[init_names]  #
            for i in range(self.nvars + N_MODEL_PARAMETERS, n_sens, self.pdo_model.nparams_sens): # fix in other branch
                y0[i] = 1

        #initialize FLAG
        FLAG = 0
        init_mass = self.external_volume * y0[3:6].sum()\
                    + DCW_TO_CELL_COUNT * self.dcw_conc(0,params) * self.cell_volume * y0[:3].sum()
        # solve ODE
        signal.alarm(100)
        time_evals = np.array([])
        sol_evals = np.array([])
        try:
            sol = solve_ivp(ds, [0, 1.1* evaluation_times[-1] * HRS_TO_SECS], y0, method="BDF", jac=ds_jac,
                            t_eval=evaluation_times * HRS_TO_SECS, atol=atol, rtol=rtol)
            fin_mass = self.external_volume * sol.y[3:6,-1].sum() + DCW_TO_CELL_COUNT*self.dcw_conc(0,params)*self.cell_volume * sol.y[:3,-1].sum()
            signal.alarm(0)
            time_evals = sol.t
            sol_evals = sol.y.T

            #check if integration is successful
            if not sol.success:
                FLAG = 3
            #check mass growth
            if fin_mass > 2*init_mass or np.any(sol.y[:6,:] < -5):
                FLAG = 4
        except ValueError:
           signal.alarm(0)
           FLAG = 1
        except TimeOutException:
            signal.alarm(0)
            FLAG = 2

        return FLAG, time_evals, sol_evals

    def _set_dcw_conc_sens(self):
        """
        Generates the derivative wrt the parameters for the dcw equation
        @return:
        """
        params_sps = self.pdo_model.params_sp
        param_dict = {param_name: param_sp for param_name, param_sp in zip(SINGLE_EXP_CALIBRATION_LIST,params_sps)}
        dcw_conc_sp = sp.Matrix([self.dcw_conc(sp.symbols('t'), param_dict)])
        dcw_conc_sp_jac = dcw_conc_sp.jacobian(params_sps)
        dcw_conc_jac_fun = sp.lambdify((sp.symbols('t'), params_sps), dcw_conc_sp_jac)
        self.dcw_conc_jac_fun = dcw_conc_jac_fun

    def get_qoi_gly_pdo_sens(self, params, evaluation_times, atol=10 ** -6, rtol=10 ** -3,
                             type = 'qoi sens'):
        """
        Integrates the DhaB-DhaT model and sensitivity equations with parameter values, param, and returns external glycerol
         1,3-PDO and cell concentration time samples, tsamples

        @param params: unstandardized parameters to evaluate differential equation
        @param evaluation_times: times to evaluate the differential equation
        @param atol: absolute tolerance
        @param rtol: relative tolerance

        @return FLAG: 0 if no issue with solving differential equation
                      1 if there was a ValueError
                      2 if there was a TimeException
                      3 if integration was unsuccessful
                      4 if mass growth
        @return qoi_gly_pdo_data: Glycerol and 1,3-PDO sampled at time samples, evaluation_times,
        (|evaluation_times| x 2 matrix)
        @return qoi_gly_pdo_sens_data: Glycerol and 1,3-PDO sampled at time samples, evaluation_times,
         (2 x |evaluation_times| x N_UNKNOWN_PARAMETERS  matrix)
        """
        #format
        calibration_params = []
        for param_name in SINGLE_EXP_CALIBRATION_LIST:
            calibration_params.append(params[param_name])

        if type == 'qoi only':
            FLAG, _, sol_concat = self.get_sol_sens_time_series(params, evaluation_times=evaluation_times, atol=atol,
                                                                rtol=rtol, type=type)
        elif type == 'qoi sens':
            FLAG, _, sol_concat = self.get_sol_sens_time_series(params, evaluation_times=evaluation_times, atol=atol,
                                                                rtol=rtol, type=type)

        if FLAG > 0:
            return FLAG, _, _

        qoi_gly_pdo = sol_concat[:, DATA_COLUMNS]

        if type == 'qoi only':
            return FLAG, qoi_gly_pdo, None
        elif type == 'qoi sens':
            # QoI sens
            qoi_gly_sens = sol_concat[:, GLY_PARAMETERS_SENS_COLUMNS]
            qoi_pdo_sens = sol_concat[:, PDO_PARAMETERS_SENS_COLUMNS]
            qoi_gly_pdo_sens = np.array([qoi_gly_sens, qoi_pdo_sens])
            return FLAG, qoi_gly_pdo, qoi_gly_pdo_sens

    def get_loglik_single_exp(self, params, evaluation_times, experimental_data_mat, sigma_exp_matrix,
                              tmatrix, atol=10**-6, rtol=10**-3, type='qoi sens'):
        """
        Computes the loglik and derivative loglik wrt parameters at parameter values, param,
        for a single experiment

        @param params: list of unstandardized parameters to evaluate differential equation
        @param evaluation_times: times to evaluate the differential equation
        @param experimental_data_mat: Glycerol and 1,3-PDO sampled at time samples, evaluation_times,
        (|evaluation_times| x  2 matrix)
        @param sigma_exp_matrix: standard deviation of Glycerol, 1,3-PDO and DCW for experimental
         evaluations at time samples, evaluation_times
          (|evaluation_times| x 2 matrix)
        @param tmatrix: diagonal matrix to transform parameters
        (N_UNKNOWN_PARAMETERS x N_UNKNOWN_PARAMETERS matrix)
        @param atol: absolute tolerance
        @param rtol: relative tolerance

        @return loglik: log-likelihood of a single experiment
        @return dloglikdparam: derivative of log-likelihood wrt params (|N_CALIBRATION_PARAMETERS| x 1  matrix)
        """
        params_dict = {}
        for param, param_name in zip(params,SINGLE_EXP_CALIBRATION_LIST):
            params_dict[param_name] = param

        init_conds = {'G_CYTO_INIT': 0., 'H_CYTO_INIT': 0., 'P_CYTO_INIT': 0.,'H_EXT_INIT': 0.,
                      'P_EXT_INIT': experimental_data_mat[0,1]}

        all_params_dict = {**params_dict,**init_conds}
        FLAG, qoi_gly_pdo, qoi_gly_pdo_sens = self.get_qoi_gly_pdo_sens(all_params_dict,
                                                                        evaluation_times=evaluation_times,
                                                                        atol = atol, rtol = rtol, type=type)

        if FLAG > 0:
            dloglikdparam = np.empty((self.pdo_model.nparams_sens,))
            dloglikdparamdparam = np.empty((self.pdo_model.nparams_sens,self.pdo_model.nparams_sens))
            dloglikdparam[:] = np.inf
            dloglikdparamdparam[:] = np.inf
            return -np.inf, dloglikdparam, dloglikdparamdparam

        # mse calculation
        qoi_obs = experimental_data_mat
        qoi_err = (qoi_gly_pdo-qoi_obs)/sigma_exp_matrix

        qoi_err = qoi_err.astype('float64')/2.
        loglik = -(qoi_err**2).sum()

        if type == 'qoi only':
            return loglik, None, None
        if type == 'qoi sens':
            #mse derivative calculation
            dloglikdparam = np.zeros((self.pdo_model.nparams_sens,))
            dloglikdparamdparam = np.zeros((self.pdo_model.nparams_sens, self.pdo_model.nparams_sens))

            for i in range(qoi_gly_pdo_sens.shape[0]):
                qoi_gly_pdo_dcw_sens_trans = np.matmul(qoi_gly_pdo_sens[i], tmatrix)
                for j in range(qoi_gly_pdo_sens.shape[1]):
                    dloglikdparam -= 2*qoi_err[j,i]*qoi_gly_pdo_dcw_sens_trans[j,:].T
                    sens_scaled = qoi_gly_pdo_dcw_sens_trans[j,:].T/(sigma_exp_matrix[j,i])
                    dloglikdparamdparam -=  np.outer(sens_scaled, sens_scaled)
            return loglik, dloglikdparam, dloglikdparamdparam