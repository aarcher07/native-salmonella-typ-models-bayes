"""
Constants parameters 

Programme written by aarcher07
Editing History:
- 1/3/21
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from os.path import dirname, abspath
from .model_constants import N_CALIBRATION_PARAMETERS,MODEL_PARAMETER_LIST, PARAMS_TO_UNITS, VARS_TO_UNITS,\
    MCP_VOLUME_SPHERICAL, AVOGADRO_CONSTANT, VARS_TO_TEX
from exp_data import INIT_CONDS_GLY_PDO_DCW
ROOT_PATH = dirname(dirname(dirname(dirname(abspath(__file__)))))

# DCW exp_data parameters
NORM_DCW_MEAN_PRIOR_PARAMETERS = pd.read_csv(ROOT_PATH + '/exp_data/data_files/dcw_fit_params', index_col = 0)
NORM_DCW_STD_PRIOR_PARAMETERS = pd.read_csv(ROOT_PATH + '/exp_data/data_files/dcw_std_params', index_col = 0)

#Uniform distribution parameters

DATA_LOG_UNIF_PARAMETER_RANGES = {'PermMCPPropanediol': np.log10([1e-7, 1e-5]),
                                  'PermMCPPropionaldehyde': np.log10([1e-9, 1e-7]),
                                  'PermMCPPropanol': np.log10([1e-9, 1e-7]),
                                  'PermMCPPropionylCoA': np.log10([1e-7, 1e-5]),
                                  'PermMCPPropionylPhosphate': np.log10([1e-7, 1e-5]),

                                  'PermCellPropanediol': np.log10([1e-6, 1e-2]),
                                  'PermCellPropionaldehyde': np.log10([1e-4, 1e-2]),
                                  'PermCellPropanol':  np.log10([1e-5, 1e-3]),
                                  'PermCellPropionylCoA': np.log10([1e-10, 1e-2]),
                                  'PermCellPropionylPhosphate': np.log10([1e-10, 1e-2]),
                                  'PermCellPropionate': np.log10([1e-4, 1e-2]),

                                  'VmaxfPduCDE': np.log10(np.array([1e4, 1e6])/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL)),
                                  'KmPduCDEPropanediol': np.log10([1e-2, 1e-1]),

                                  'VmaxfPduQ': np.log10(np.array([1e4, 1e6])/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL)),
                                  'KmPduQPropionaldehyde': np.log10([1e1, 1e2]),
                                  'VmaxrPduQ': np.log10(np.array([1e3, 1e5]))/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL),
                                  'KmPduQPropanol': np.log10([1e1, 1e3]),

                                  'VmaxfPduP': np.log10(np.array([1e3, 1e5]))/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL), #kcatf ~ 1000
                                  'KmPduPPropionaldehyde': np.log10([0.1, 10]),
                                  'VmaxrPduP': np.log10(np.array([1e3, 1e5]))/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL), #kcatr ~ 100
                                  'KmPduPPropionylCoA': np.log10([1e-3, 1e3]),

                                  'VmaxfPduL': np.log10(np.array([1e1, 1e2]))/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL), # not studied can use thermo
                                  'KmPduLPropionylCoA': np.log10([1e-3, 1e3]), #not studied
                                  'VmaxrPduL': np.log10(np.array([1e3, 1e5]))/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL), #kcatr ~ 20.7
                                  'KmPduQPropionylPhosphate': np.log10([1e-1, 1e1]),

                                  'VmaxfPduW': np.log10([1e3, 1e5]), #Doesnt look like much available
                                  'KmPduWPropionylPhosphate': np.log10([1e-1, 1e1]),#Doesnt look like much available

                                  'VmaxfPrpE': np.log10([1e3, 1e5]), #TODO: put in actual values
                                  'KmPrpEPropionate': np.log10([1e-1, 1e1]), #TODO: put in actual values

                                  'VmaxfPrpC': np.log10([1e-1, 1e1]), #TODO: put in actual values
                                  'KmPrpCPropionylCoA': np.log10([1e1 * 0.1, 1e2 * 10]), #TODO: put in actual values
                                  }

# Normal model distribution parameters
LOG_NORM_MODEL_PRIOR_MEAN = {param_name: np.mean(DATA_LOG_UNIF_PARAMETER_RANGES[param_name])
                             for param_name in MODEL_PARAMETER_LIST}
LOG_NORM_MODEL_PRIOR_STD = {param_name: (DATA_LOG_UNIF_PARAMETER_RANGES[param_name][1]
                                         -np.mean(DATA_LOG_UNIF_PARAMETER_RANGES[param_name]))/2
                            for param_name in MODEL_PARAMETER_LIST}

LOG_NORM_MODEL_PRIOR_PARAMETERS = {param_name: [LOG_NORM_MODEL_PRIOR_MEAN[param_name],
                                                LOG_NORM_MODEL_PRIOR_STD[param_name]]
                                   for param_name in MODEL_PARAMETER_LIST}

# Glycerol model distribution parameters
NORM_EXT_INIT_PRIOR_PARAMETERS = {'PROPANEDIOL_EXT_INIT': [INIT_CONDS[0, 0], 5.],
                                  }

NORM_EXT_INIT_PRIOR_MEAN = {param_name: NORM_EXT_INIT_PRIOR_PARAMETERS[param_name][0]
                            for param_name in NORM_EXT_INIT_PRIOR_PARAMETERS.keys()}

NORM_EXT_INIT_PRIOR_STD = {param_name: NORM_EXT_INIT_PRIOR_PARAMETERS[param_name][1]
                           for param_name in NORM_EXT_INIT_PRIOR_PARAMETERS.keys()}

# prior parameters for single experiment
NORM_PRIOR_MEAN_SINGLE_EXP = np.array([*list(LOG_NORM_MODEL_PRIOR_MEAN.values()),
                                       NORM_EXT_INIT_PRIOR_MEAN['PROPANEDIOL_EXT_INIT']])
NORM_PRIOR_STD_RT_SINGLE_EXP = np.diag([*list(LOG_NORM_MODEL_PRIOR_STD.values()),
                                        NORM_EXT_INIT_PRIOR_STD['PROPANEDIOL_EXT_INIT']])

# prior parameters for all experiment
NORM_PRIOR_PARAMETER_ALL_EXP_DICT = {**LOG_NORM_MODEL_PRIOR_PARAMETERS, **NORM_EXT_INIT_PRIOR_PARAMETERS}

NORM_PRIOR_MEAN_ALL_EXP = np.array([*list(LOG_NORM_MODEL_PRIOR_MEAN.values()),
                                    *list(NORM_EXT_INIT_PRIOR_MEAN.values())])


NORM_PRIOR_STD_RT_ALL_EXP = np.diag([*list(LOG_NORM_MODEL_PRIOR_STD.values()),
                                     *list(NORM_EXT_INIT_PRIOR_STD.values())])


# teX names for parameters
EXT_INIT_TO_TEX = {'PROPANEDIOL_EXT_INIT': "$P(0)$ for first experiment"}

VARS_ALL_EXP_TO_TEX = {**VARS_TO_TEX,
                       **EXT_INIT_TO_TEX}

G_EXT_INIT_TO_TEX = {'PROPANEDIOL_EXT_INIT': "mM"}

VARS_ALL_EXP_TO_UNITS = {**VARS_TO_UNITS, **G_EXT_INIT_TO_TEX}
