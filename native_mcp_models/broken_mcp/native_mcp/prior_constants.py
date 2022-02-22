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
ROOT_PATH = dirname(dirname(dirname(dirname(abspath(__file__)))))

#Uniform distribution parameters

DATA_LOG_UNIF_PARAMETER_RANGES = {'PermCellPropanediol': np.log10([1e-6, 1e-2]),
                                  'PermCellPropionaldehyde': np.log10([1e-4, 1e-2]),
                                  'PermCellPropanol':  np.log10([1e-5, 1e-3]),
                                  'PermCellPropionylCoA': np.log10([1e-10, 1e-2]),
                                  'PermCellPropionylPhosphate': np.log10([1e-10, 1e-2]),
                                  'PermCellPropionate': np.log10([1e-4, 1e-2]),

                                  'VmaxfPduCDE': np.log10(np.array([1e3, 1e7])/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL)),
                                  'KmPduCDEPropanediol': np.log10([1e-2, 1e-1]),

                                  'VmaxfPduQ': np.log10(np.array([1e3, 1e6])/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL)),
                                  'KmPduQPropionaldehyde': np.log10([1e-2, 1e3]),
                                  'VmaxrPduQ': np.log10(np.array([1e2, 1e5])/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL)),
                                  'KmPduQPropanol': np.log10([1e0, 1e4]),

                                  'VmaxfPduP': np.log10(np.array([1e3, 1e8])/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL)), #kcatf ~ 1000
                                  'KmPduPPropionaldehyde': np.log10([0.1, 10]),
                                  'VmaxrPduP': np.log10(np.array([1e2, 1e7])/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL)), #kcatr ~ 100
                                  'KmPduPPropionylCoA': np.log10([1e-2, 1e3]),

                                  'VmaxfPduL': np.log10(np.array([1e1, 1e4])/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL)), # not studied can use thermo
                                  'KmPduLPropionylCoA': np.log10([1e-3, 1e3]), #not studied
                                  'VmaxrPduL': np.log10(np.array([1e2, 1e6])/(AVOGADRO_CONSTANT * MCP_VOLUME_SPHERICAL)), #kcatr ~ 20.7
                                  'KmPduQPropionylPhosphate': np.log10([1e-3, 1e0]),

                                  'VmaxfPduW': np.log10([1e0, 1e3]), #Doesnt look like much available
                                  'KmPduWPropionylPhosphate': np.log10([1e-3, 1e3]),#Doesnt look like much available

                                  'VmaxfPrpE': np.log10([1e-2, 1e3]),
                                  'KmPrpEPropionate': np.log10([1e-3, 1e0]),

                                  'VmaxfPrpC': np.log10([1e-1, 1e3]),
                                  'KmPrpCPropionylCoA': np.log10([1e-4, 1e0])
                                  }
MCP_PARAMETER_RANGES = {'PermMCPPropanediol': np.log10([1e-7, 1e-5]),
                        'PermMCPPropionaldehyde': np.log10([1e-9, 1e-7]),
                        'PermMCPPropanol': np.log10([1e-9, 1e-7]),
                        'PermMCPPropionylCoA': np.log10([1e-7, 1e-5]),
                        'PermMCPPropionylPhosphate': np.log10([1e-7, 1e-5]),
                        'NPduCDE': [],
                        'NPduQ': [],
                        'PduL': [],
                        'nmcps': [3,30]}

bMCP_PARAMETER_RANGES = {'PermPolarBody'
                         'NPduCDE': [],
                         'NPduQ': [],
                         'PduL': [],
                         'nmcps': [3,30]}

# Normal model distribution parameters
LOG_NORM_MODEL_PRIOR_MEAN = {param_name: np.mean(DATA_LOG_UNIF_PARAMETER_RANGES[param_name])
                             for param_name in MODEL_PARAMETER_LIST}
LOG_NORM_MODEL_PRIOR_STD = {param_name: (DATA_LOG_UNIF_PARAMETER_RANGES[param_name][1]
                                         -np.mean(DATA_LOG_UNIF_PARAMETER_RANGES[param_name]))/2
                            for param_name in MODEL_PARAMETER_LIST}

LOG_NORM_MODEL_PRIOR_PARAMETERS = {param_name: [LOG_NORM_MODEL_PRIOR_MEAN[param_name],
                                                LOG_NORM_MODEL_PRIOR_STD[param_name]]
                                   for param_name in MODEL_PARAMETER_LIST}

# prior parameters for single experiment
NORM_PRIOR_MEAN_SINGLE_EXP = {}
NORM_PRIOR_STD_RT_SINGLE_EXP = {}
for gly_cond in ['WT']:
    NORM_PRIOR_MEAN_SINGLE_EXP[gly_cond] = np.array([*list(LOG_NORM_MODEL_PRIOR_MEAN.values()),])
    NORM_PRIOR_STD_RT_SINGLE_EXP[gly_cond] = np.diag([*list(LOG_NORM_MODEL_PRIOR_STD.values())])


