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
from .pdu_mcp_model import pdu_mcp_model
import multiprocessing as mp
import time

class TimeOutException(Exception):
    pass


def handler(signum, frame):
    raise TimeOutException("Integration is taking too long")


signal.signal(signal.SIGALRM, handler)


class pdu_mcp_model_log(pdu_mcp_model):

    def __init__(self, mcp_volume, mcp_surface_area, cell_conc_od,
                 cell_volume=CELL_VOLUME, cell_surface_area=CELL_SURFACE_AREA,
                 external_volume=EXTERNAL_VOLUME):
        """
        Initializes parameters to be used numerial scheme
        :param cell_volume: cell volume in m^3
        :param cell_surface_area: cell surface area in m^2
        :param external_volume: external volume containing cells in metres^3
        """
        # geometric parameters
        super().__init__(mcp_volume, mcp_surface_area, cell_conc_od,
                         cell_volume, cell_surface_area, external_volume)
        # integration parameters
        self.mcp_model = LSAModelBuilder(self._RHS, self.nvars, self.nparams, list(range(N_MODEL_PARAMETERS + 1)))

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
                params.append(10 ** param_val)
            else:
                params.append(param_val)

        d = super()._RHS(t, x, params)
        return d

    def get_sol_sens_time_series(self, params, evaluation_times=None, atol=10 ** -6, rtol=10 ** -3,
                                 type='qoi sens'):
        """
        Integrates the Pdu MCP pathway model and sensitivity equation with parameter values, param, and returns e
        time series samples at tsamples

        @param params: dictionary of unstandardized parameters to evaluate differential equation
        @param evaluation_times: time samples
        @param atol: absolute tolerance
        @param rtol: relative tolerance

        @return FLAG: 0 if no issue with solving differential equation
                      1 if there was a ValueError
                      2 if there was a TimeException
                      3 if integration was unsuccessful
                      4 if mass growth
        @return t_concat: times at which the differential equations were solved
        @return sol_conat: differential equations solutions sampled at time samples, evaluation_times (17 (17 + 17*self.n_params_sens)x |evaluation_times| matrix)
        """
        # format inputs
        calibration_params = []
        for param_name in SINGLE_EXP_CALIBRATION_LIST:
            calibration_params.append(params[param_name])

        # ODE solver args initialization
        if type == 'qoi only':
            ds = lambda t, x: self.mcp_model.RHS(t, x, calibration_params)
            ds_jac = lambda t, x: self.mcp_model.RHS_jac_state_fun(t, x, calibration_params)

            # initial conditions
            y0 = np.zeros(N_VARIABLES)
            for i, init_names in enumerate(VARIABLE_INIT_NAMES):
                y0[i] = params[init_names]

        elif type == 'qoi sens':
            ds = lambda t, x: self.mcp_model.dsens(t, x, calibration_params)
            ds_jac = lambda t, x: self.mcp_model.dsens_jac(t, x, calibration_params)

            # initial conditions
            n_sens = self.nvars + self.nvars * self.mcp_model.nparams_sens
            y0 = np.zeros(n_sens)
            for i, init_names in enumerate(VARIABLE_INIT_NAMES):
                y0[i] = params[init_names]  #

        # initialize FLAG
        FLAG = 0
        print(OD_TO_CELL_COUNT * self.cell_conc_od(0) * self.cell_volume * y0[5:11].sum())

        init_mass = OD_TO_CELL_COUNT * self.cell_conc_od(0) * params['nmcps'] * y0[:5].sum() \
                    + OD_TO_CELL_COUNT * self.cell_conc_od(0) * self.cell_volume * y0[5:11].sum() \
                    + self.external_volume * y0[11:17].sum()
        print('init mass: ' + str(init_mass))

        # solve ODE
        signal.alarm(500)
        time_evals = np.array([])
        sol_evals = np.array([])
        try:
            time1 = time.time()
            sol = solve_ivp(ds, [0, 1.1 * evaluation_times[-1] * HRS_TO_SECS], y0, method="BDF", jac=ds_jac,
                            t_eval=evaluation_times * HRS_TO_SECS, atol=atol, rtol=rtol)
            time2 = time.time()
            fin_mass = OD_TO_CELL_COUNT * self.cell_conc_od(0) * params['nmcps'] * sol.y[:5, -1].sum() \
                       + OD_TO_CELL_COUNT * self.cell_conc_od(0) * self.cell_volume * sol.y[5:11, -1].sum() \
                       + self.external_volume * sol.y[11:17, -1].sum()
            signal.alarm(0)
            print('final mass: ' + str(fin_mass))
            print("sol time: " + str((time2-time1)/60))

            time_evals = sol.t
            sol_evals = sol.y.T

            # check if integration is successful
            if not sol.success:
                FLAG = 3
            # check mass growth
            if fin_mass > 2 * init_mass or np.any(sol.y[:N_VARIABLES,:] < -5):
                FLAG = 4
        except ValueError:
            signal.alarm(0)
            FLAG = 1
        except TimeOutException:
            signal.alarm(0)
            FLAG = 2

        return FLAG, time_evals, sol_evals

    def get_sol_sens(self, params, evaluation_times, atol=10 ** -6, rtol=10 ** -3,
                             type='qoi sens'):
        """
        Integrates the Pdu model and sensitivity equations with parameter values, param, and returns external 1,2-PD, Propionaldehyde,
         Propanol and Propionate time samples, tsamples

        @param params: unstandardized parameters to evaluate differential equation
        @param evaluation_times: times to evaluate the differential equation
        @param atol: absolute tolerance
        @param rtol: relative tolerance

        @return FLAG: 0 if no issue with solving differential equation
                      1 if there was a ValueError
                      2 if there was a TimeException
                      3 if integration was unsuccessful
                      4 if mass growth
        @return qoi: 1,2-PD, Propionaldehyde, Propanol and Propionate sampled at time samples, evaluation_times,
        (|evaluation_times| x 4 matrix)
        @return qoi_sens: 1,2-PD, Propionaldehyde, Propanol and Propionate senstivities sampled at time samples, evaluation_times,
         4 x |evaluation_times| x (N_MODEL_PARAMETERS + 1)  matrix)
        """
        # format
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

        qoi = sol_concat[:, DATA_COLUMNS]

        if type == 'qoi only':
            return FLAG, qoi, None
        elif type == 'qoi sens':
            # QoI sens
            qoi_propanediol_sens = sol_concat[:, PROPANEDIOL_PARAMETERS_SENS_COLUMNS]
            qoi_propionaldehyde_sens = sol_concat[:, PROPIONALDEHYDE_PARAMETERS_SENS_COLUMNS]
            qoi_propanol_sens = sol_concat[:, PROPANOL_PARAMETERS_SENS_COLUMNS]
            qoi_propionate_sens = sol_concat[:, PROPIONATE_PARAMETERS_SENS_COLUMNS]
            qoi_sens = np.array(
                [qoi_propanediol_sens, qoi_propionaldehyde_sens, qoi_propanol_sens, qoi_propionate_sens])
            return FLAG, qoi, qoi_sens

    def get_loglik_single_exp(self, params, evaluation_times, experimental_data_mat, sigma_exp_matrix,
                              tmatrix, atol=10 ** -6, rtol=10 ** -3, type='qoi sens'):
        """
        Computes the loglik and derivative loglik wrt parameters at parameter values, param,
        for a single experiment

        @param params: list of unstandardized parameters to evaluate differential equation
        @param evaluation_times: times to evaluate the differential equation
        @param experimental_data_mat: 1,2-PD, Propionaldehyde, Propanol and Propionate sampled at time samples, evaluation_times,
        (|evaluation_times| x  4 matrix)
        @param sigma_exp_matrix: standard deviation of 1,2-PD, Propionaldehyde, Propanol and Propionate for experimental
         evaluations at time samples, evaluation_times
          (|evaluation_times| x 4 matrix)
        @param tmatrix: diagonal matrix to transform parameters
        @param atol: absolute tolerance
        @param rtol: relative tolerance

        @return loglik: log-likelihood of a single experiment
        @return dloglikdparam: derivative of log-likelihood wrt params (|N_CALIBRATION_PARAMETERS| x 1  matrix)
        @return dloglikdparam: derivative of log-likelihood wrt params (|N_CALIBRATION_PARAMETERS| x |N_CALIBRATION_PARAMETERS}  matrix)

        """
        params_dict = {}
        for param, param_name in zip(params, SINGLE_EXP_CALIBRATION_LIST):
            params_dict[param_name] = param

        init_conds = {'PROPANEDIOL_MCP_INIT': experimental_data_mat[0, 0],
                      'PROPIONALDEHYDE_MCP_INIT': experimental_data_mat[0, 1],
                      'PROPANOL_MCP_INIT': experimental_data_mat[0, 2],
                      'PROPIONYLCOA_MCP_INIT': 0,
                      'PROPIONYLPHOSPHATE_MCP_INIT': experimental_data_mat[0, 3],

                      'PROPANEDIOL_CYTO_INIT': experimental_data_mat[0, 0],
                      'PROPIONALDEHYDE_CYTO_INIT': experimental_data_mat[0, 1],
                      'PROPANOL_CYTO_INIT': experimental_data_mat[0, 2],
                      'PROPIONYLCOA_CYTO_INIT': 0,
                      'PROPIONYLPHOSPHATE_CYTO_INIT': 0,
                      'PROPIONATE_CYTO_INIT': experimental_data_mat[0, 3],

                      'PROPANEDIOL_EXT_INIT': experimental_data_mat[0, 0],
                      'PROPIONALDEHYDE_EXT_INIT': experimental_data_mat[0, 1],
                      'PROPANOL_EXT_INIT': experimental_data_mat[0, 2],
                      'PROPIONYLCOA_EXT_INIT': 0,
                      'PROPIONYLPHOSPHATE_EXT_INIT': 0,
                      'PROPIONATE_EXT_INIT': experimental_data_mat[0, 3]}

        all_params_dict = {**params_dict, **init_conds}
        FLAG, qoi, qoi_sens = self.get_qoi_gly_pdo_sens(all_params_dict,
                                                        evaluation_times=evaluation_times,
                                                        atol=atol, rtol=rtol, type=type)

        if FLAG > 0:
            dloglikdparam = np.empty((self.mcp_model.nparams_sens,))
            dloglikdparamdparam = np.empty((self.mcp_model.nparams_sens, self.mcp_model.nparams_sens))
            dloglikdparam[:] = np.inf
            dloglikdparamdparam[:] = np.inf
            return -np.inf, dloglikdparam, dloglikdparamdparam

        # mse calculation
        qoi_obs = experimental_data_mat
        qoi_err = (qoi - qoi_obs) / sigma_exp_matrix

        qoi_err = qoi_err.astype('float64') / 2.
        loglik = -(qoi_err ** 2).sum()

        if type == 'qoi only':
            return loglik, None, None
        if type == 'qoi sens':
            # mse derivative calculation
            dloglikdparam = np.zeros((self.mcp_model.nparams_sens,))
            dloglikdparamdparam = np.zeros((self.mcp_model.nparams_sens, self.mcp_model.nparams_sens))

            for i in range(qoi_sens.shape[0]):
                qoi_sens_trans = np.matmul(qoi_sens[i], tmatrix)
                for j in range(qoi_sens.shape[1]):
                    dloglikdparam -= 2 * qoi_err[j, i] * qoi_sens_trans[j, :].T
                    sens_scaled = qoi_sens_trans[j, :].T / (sigma_exp_matrix[j, i])
                    dloglikdparamdparam -= np.outer(sens_scaled, sens_scaled)
            return loglik, dloglikdparam, dloglikdparamdparam
