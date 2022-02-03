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
from .model_constants import N_CALIBRATION_PARAMETERS,MODEL_PARAMETER_LIST, VARS_TO_TEX, VARS_TO_UNITS
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

                                  'VmaxfPduCDE': np.log10([1e4, 1e6]),
                                  'KmPduCDEPropanediol': np.log10([1e-2, 1e-1]),

                                  'VmaxfPduQ': np.array([1e4, 1e6]),
                                  'KmPduQPropionaldehyde': np.log10([1e1, 1e2]),
                                  'VmaxrPduQ': np.log10([1e3, 1e5]),
                                  'KmPduQPropanol': np.log10([1e1, 1e3]),

                                  'VmaxfPduP': np.array([np.nan, np.log10(1e-2)]),
                                  'KmPduPPropionaldehyde': np.log10([1e2 * 0.1, 1e3 * 10]),
                                  'VmaxrPduP': np.log10([1e-1, 1e1]),
                                  'KmPduPPropionylCoA': np.log10([1e1 * 0.1, 1e2 * 10]),

                                  'VmaxfPduL': np.array([np.nan, np.log10(1e-2)]),
                                  'KmPduLPropionylCoA': np.log10([1e2 * 0.1, 1e3 * 10]),
                                  'VmaxrPduL': np.log10([1e-1, 1e1]),
                                  'KmPduQPropionylPhosphate': np.log10([1e1 * 0.1, 1e2 * 10]),

                                  'VmaxfPduW': np.array([np.nan, np.log10(1e-2)]), #Doesnt look like much available
                                  'KmPduWPropionylPhosphate': np.log10([1e2 * 0.1, 1e3 * 10]),#Doesnt look like much available

                                  'VmaxfPrpE': np.log10([1e-1, 1e1]),
                                  'KmPduQPropionate': np.log10([1e1 * 0.1, 1e2 * 10]),

                                  'VmaxfPrpC': np.log10([1e-1, 1e1]),
                                  'KmPduQPropionylCoA': np.log10([1e1 * 0.1, 1e2 * 10]),
                                  }

# Normal model distribution parameters
LOG_NORM_MODEL_PRIOR_MEAN = {param_name: np.mean(DATA_LOG_UNIF_PARAMETER_RANGES[param_name])
                             for param_name in MODEL_PARAMETER_LIST}
LOG_NORM_MODEL_PRIOR_MEAN['PermCell3HPA'] = -4
LOG_NORM_MODEL_PRIOR_STD = {param_name: (DATA_LOG_UNIF_PARAMETER_RANGES[param_name][1]
                                         -np.mean(DATA_LOG_UNIF_PARAMETER_RANGES[param_name]))/2
                            for param_name in MODEL_PARAMETER_LIST}

LOG_NORM_MODEL_PRIOR_STD['PermCell3HPA'] = (DATA_LOG_UNIF_PARAMETER_RANGES['PermCell3HPA'][1]+4)/2

LOG_NORM_MODEL_PRIOR_PARAMETERS = {param_name: [LOG_NORM_MODEL_PRIOR_MEAN[param_name],
                                                LOG_NORM_MODEL_PRIOR_STD[param_name]]
                                   for param_name in MODEL_PARAMETER_LIST}

# Glycerol model distribution parameters
NORM_G_EXT_INIT_PRIOR_PARAMETERS = {'G_EXT_INIT_50': [INIT_CONDS_GLY_PDO_DCW[50][0], 15.],
                                    'G_EXT_INIT_60': [INIT_CONDS_GLY_PDO_DCW[60][0], 15.],
                                    'G_EXT_INIT_70': [INIT_CONDS_GLY_PDO_DCW[70][0], 15.],
                                    'G_EXT_INIT_80': [INIT_CONDS_GLY_PDO_DCW[80][0], 15.]
                                    }

NORM_G_EXT_INIT_PRIOR_MEAN = {param_name: NORM_G_EXT_INIT_PRIOR_PARAMETERS[param_name][0]
                              for param_name in NORM_G_EXT_INIT_PRIOR_PARAMETERS.keys()}

NORM_G_EXT_INIT_PRIOR_STD = {param_name: NORM_G_EXT_INIT_PRIOR_PARAMETERS[param_name][1]
                             for param_name in NORM_G_EXT_INIT_PRIOR_PARAMETERS.keys()}

# DCW model distribution parameters

NORM_DCW_PRIOR_PARAMETERS_50 = {param_name + '_50': [mean, std] for param_name, mean, std in zip(NORM_DCW_MEAN_PRIOR_PARAMETERS.columns,
                                                                                                 NORM_DCW_MEAN_PRIOR_PARAMETERS.loc[50, :],
                                                                                                 NORM_DCW_STD_PRIOR_PARAMETERS.loc[50, :])}

NORM_DCW_PRIOR_PARAMETERS_60 = {param_name + '_60': [mean, std] for param_name, mean, std in zip(NORM_DCW_MEAN_PRIOR_PARAMETERS.columns,
                                                                                                 NORM_DCW_MEAN_PRIOR_PARAMETERS.loc[60, :],
                                                                                                 NORM_DCW_STD_PRIOR_PARAMETERS.loc[60, :])}

NORM_DCW_PRIOR_PARAMETERS_70 = {param_name + '_70': [mean, std] for param_name, mean, std in zip(NORM_DCW_MEAN_PRIOR_PARAMETERS.columns,
                                                                                                 NORM_DCW_MEAN_PRIOR_PARAMETERS.loc[70, :],
                                                                                                 NORM_DCW_STD_PRIOR_PARAMETERS.loc[70, :])}

NORM_DCW_PRIOR_PARAMETERS_80 = {param_name + '_80': [mean, std] for param_name, mean, std in zip(NORM_DCW_MEAN_PRIOR_PARAMETERS.columns,
                                                                                                 NORM_DCW_MEAN_PRIOR_PARAMETERS.loc[80, :],
                                                                                                 NORM_DCW_STD_PRIOR_PARAMETERS.loc[80, :])}

# prior parameters for single experiment
NORM_PRIOR_MEAN_SINGLE_EXP = {}
NORM_PRIOR_STD_RT_SINGLE_EXP = {}
for gly_cond in [50,60,70,80]:
    NORM_PRIOR_MEAN_SINGLE_EXP[gly_cond] = np.array([*list(LOG_NORM_MODEL_PRIOR_MEAN.values()),
                                                     NORM_G_EXT_INIT_PRIOR_MEAN['G_EXT_INIT_' + str(gly_cond)],
                                                     *NORM_DCW_MEAN_PRIOR_PARAMETERS.loc[gly_cond, :].tolist()])
    NORM_PRIOR_STD_RT_SINGLE_EXP[gly_cond] = np.diag([*list(LOG_NORM_MODEL_PRIOR_STD.values()),
                                                      NORM_G_EXT_INIT_PRIOR_STD['G_EXT_INIT_' + str(gly_cond)],
                                                      *NORM_DCW_STD_PRIOR_PARAMETERS.loc[gly_cond, :].tolist()])

# prior parameters for all experiment
NORM_PRIOR_PARAMETER_ALL_EXP_DICT = {**LOG_NORM_MODEL_PRIOR_PARAMETERS, **NORM_G_EXT_INIT_PRIOR_PARAMETERS,
                                     **NORM_DCW_PRIOR_PARAMETERS_50,  **NORM_DCW_PRIOR_PARAMETERS_60,
                                     **NORM_DCW_PRIOR_PARAMETERS_70, **NORM_DCW_PRIOR_PARAMETERS_80}

NORM_PRIOR_MEAN_ALL_EXP = np.array([*list(LOG_NORM_MODEL_PRIOR_MEAN.values()),
                                    *list(NORM_G_EXT_INIT_PRIOR_MEAN.values()),
                                    *NORM_DCW_MEAN_PRIOR_PARAMETERS.loc[50, :].tolist(),
                                    *NORM_DCW_MEAN_PRIOR_PARAMETERS.loc[60, :].tolist(),
                                    *NORM_DCW_MEAN_PRIOR_PARAMETERS.loc[70, :].tolist(),
                                    *NORM_DCW_MEAN_PRIOR_PARAMETERS.loc[80, :].tolist()])


NORM_PRIOR_STD_RT_ALL_EXP = np.diag([*list(LOG_NORM_MODEL_PRIOR_STD.values()),
                                     *list(NORM_G_EXT_INIT_PRIOR_STD.values()),
                                     *NORM_DCW_STD_PRIOR_PARAMETERS.loc[50, :].tolist(),
                                     *NORM_DCW_STD_PRIOR_PARAMETERS.loc[60, :].tolist(),
                                     *NORM_DCW_STD_PRIOR_PARAMETERS.loc[70, :].tolist(),
                                     *NORM_DCW_STD_PRIOR_PARAMETERS.loc[80, :].tolist()])


# teX names for parameters

G_EXT_INIT_TO_TEX = {'G_EXT_INIT_50': "$G(0)$ for first experiment",
                     'G_EXT_INIT_60': "$G(0)$ for second experiment",
                     'G_EXT_INIT_70': "$G(0)$ for third experiment",
                     'G_EXT_INIT_80': "$G(0)$ for fourth experiment"
                     }

DCW_TO_TEX_50 = {param_name + "_50": "$"+param_name + "_{50}$" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}
DCW_TO_TEX_60 = {param_name + "_60": "$"+param_name + "_{60}$" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}
DCW_TO_TEX_70 = {param_name + "_70": "$"+param_name + "_{70}$" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}
DCW_TO_TEX_80 = {param_name + "_80": "$"+param_name + "_{80}$" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}

VARS_ALL_EXP_TO_TEX = {**VARS_TO_TEX, **G_EXT_INIT_TO_TEX, **DCW_TO_TEX_50, **DCW_TO_TEX_60, **DCW_TO_TEX_70,
                       **DCW_TO_TEX_80}

G_EXT_INIT_TO_TEX = {'G_EXT_INIT_50': "mM",
                     'G_EXT_INIT_60': "mM",
                     'G_EXT_INIT_70': "mM",
                     'G_EXT_INIT_80': "mM"
                     }
DCW_TO_UNITS_50 = {param_name + "_50": "" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}
DCW_TO_UNITS_60 = {param_name + "_60": "" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}
DCW_TO_UNITS_70 = {param_name + "_70": "" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}
DCW_TO_UNITS_80 = {param_name + "_80": "" for param_name in NORM_DCW_MEAN_PRIOR_PARAMETERS.columns}

VARS_ALL_EXP_TO_UNITS = {**VARS_TO_UNITS, **G_EXT_INIT_TO_TEX, **DCW_TO_UNITS_50, **DCW_TO_UNITS_60, **DCW_TO_UNITS_70,
                         **DCW_TO_UNITS_80}
