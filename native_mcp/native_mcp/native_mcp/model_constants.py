"""
Model constants

Programme written by aarcher07
Editing History:
- 1/3/21
"""

import numpy as np
import pandas as pd

########################################################################################################################
############################################ MODEL PARAMETER CONSTANTS #################################################
########################################################################################################################

MODEL_PARAMETER_LIST = ['PermMCPPropanediol', 'PermMCPPropionaldehyde', 'PermMCPPropanol', 'PermMCPPropionylCoA',
                        'PermMCPPropionylPhosphate',

                        'PermCellPropanediol', 'PermCellPropionaldehyde', 'PermCellPropanol', 'PermCellPropionylCoA',
                        'PermCellPropionylPhosphate', 'PermCellPropionate',


                        'VmaxfPduCDE', 'KmPduCDEPropanediol',
                        'VmaxfPduQ', 'KmPduQPropionaldehyde',
                        'VmaxrPduQ', 'KmPduQPropanol',

                        'VmaxfPduP', 'KmPduPPropionaldehyde',
                        'VmaxrPduP', 'KmPduPPropionylCoA',

                        'VmaxfPduL', 'KmPduLPropionylCoA',
                        'VmaxrPduL', 'KmPduQPropionylPhosphate',

                        'VmaxfPduW', 'KmPduWPropionylPhosphate',
                        'VmaxfPrpE', 'KmPrpEPropionate',
                        'VmaxrfPrpC', 'KmPrpCPropionylCoA',

                        'nmcps'
                        ]

SINGLE_EXP_CALIBRATION_LIST = [*MODEL_PARAMETER_LIST,

                               'PROPANEDIOL_EXT_INIT', 'PROPIONALDEHYDE_EXT_INIT', 'PROPANOL_EXT_INIT',
                               'PROPIONYLCOA_EXT_INIT', 'PROPIONYLPHOSPHATE_EXT_INIT',
                               'PROPIONATE_EXT_INIT'
                               ]

VARIABLE_INIT_NAMES = ['PROPANEDIOL_MCP_INIT', 'PROPIONALDEHYDE_MCP_INIT', 'PROPANOL_MCP_INIT', 'PROPIONYLCOA_MCP_INIT',
                       'PROPIONYLPHOSPHATE_MCP_INIT',

                       'PROPANEDIOL_CYTO_INIT', 'PROPIONALDEHYDE_CYTO_INIT', 'PROPANOL_CYTO_INIT',
                       'PROPIONYLCOA_CYTO_INIT',
                       'PROPIONYLPHOSPHATE_CYTO_INIT', 'PROPIONATE_CYTO_INIT'

                       'PROPANEDIOL_EXT_INIT', 'PROPIONALDEHYDE_EXT_INIT',
                       'PROPANOL_EXT_INIT', 'PROPIONYLCOA_EXT_INIT',
                       'PROPIONYLPHOSPHATE_EXT_INIT', 'PROPIONATE_EXT_INIT']

N_VARIABLES = 17
N_MODEL_PARAMETERS = len(MODEL_PARAMETER_LIST)
N_CALIBRATION_PARAMETERS = len(SINGLE_EXP_CALIBRATION_LIST)
########################################################################################################################
############################################### CONVERSION CONSTANTS ###################################################
########################################################################################################################

HRS_TO_SECS = 60 * 60
# DCW to CELL CONCENTRATION
OD_TO_CELL_CONCENTRATION = 1e15  # number of cell/m^3 per 1 OD

#######################################################################################################################
############################################ CELL GEOMETRY CONSTANTS ##################################################
#######################################################################################################################

CELL_RADIUS = 0.375e-6
CELL_LENGTH = 2.47e-6
MCP_RADIUS = 7e-8
CELL_SURFACE_AREA = 2 * np.pi * CELL_RADIUS * CELL_LENGTH
CELL_VOLUME = 4 * np.pi / 3 * (CELL_RADIUS) ** 3 + np.pi * (CELL_LENGTH - 2 * CELL_RADIUS) * (CELL_RADIUS ** 2)
MCP_VOLUME_SPHERICAL = (4 / 3) * np.pi * (MCP_RADIUS ** 3)
MCP_SURFACE_AREA_SPHERICAL = 4 * np.pi * (MCP_RADIUS ** 2)

#######################################################################################################################
############################################### PARAMETER CONSTANTS ###################################################
#######################################################################################################################

DATA_COLUMNS = [11, 12, 13, 16]  # indices of QoI in the differential equation
PROPANEDIOL_PARAMETERS_SENS_COLUMNS = np.arange(N_VARIABLES + DATA_COLUMNS[0] * N_CALIBRATION_PARAMETERS,
                                                N_VARIABLES + (DATA_COLUMNS[0] + 1) * N_CALIBRATION_PARAMETERS)

PROPIONALDEHYDE_PARAMETERS_SENS_COLUMNS = np.arange(N_VARIABLES + DATA_COLUMNS[1] * N_CALIBRATION_PARAMETERS,
                                                    N_VARIABLES + (DATA_COLUMNS[1] + 1) * N_CALIBRATION_PARAMETERS)

PROPANOL_PARAMETERS_SENS_COLUMNS = np.arange(N_VARIABLES + DATA_COLUMNS[2] * N_CALIBRATION_PARAMETERS,
                                             N_VARIABLES + (DATA_COLUMNS[2] + 1) * N_CALIBRATION_PARAMETERS)

PROPIONATE_PARAMETERS_SENS_COLUMNS = np.arange(N_VARIABLES + DATA_COLUMNS[3] * N_CALIBRATION_PARAMETERS,
                                               N_VARIABLES + (DATA_COLUMNS[3] + 1) * N_CALIBRATION_PARAMETERS)
#######################################################################################################################
#################################################### MISC CONSTANTS ###################################################
#######################################################################################################################

EXTERNAL_VOLUME = 5e-5  # external volume from experiment
OD_TO_CELL_COUNT = OD_TO_CELL_CONCENTRATION * EXTERNAL_VOLUME

########################################################################################################################
############################################## FORMATTING CONSTANTS ####################################################
########################################################################################################################


MODEL_PARAMETER_LIST_UNITS = {'PermMCPPropanediol': 'm/s',
                              'PermMCPPropionaldehyde': 'm/s',
                              'PermMCPPropanol': 'm/s',
                              'PermMCPPropionylCoA': 'm/s',
                              'PermMCPPropionylPhosphate': 'm/s',

                              'PermCellPropanediol': 'm/s',
                              'PermCellPropionaldehyde': 'm/s',
                              'PermCellPropanol': 'm/s',
                              'PermCellPropionylCoA': 'm/s',
                              'PermCellPropionylPhosphate': 'm/s',
                              'PermCellPropionate': 'm/s',

                              'VmaxfPduCDE': "mM/s",
                              'KmPduCDEPropanediol': "mM",

                              'VmaxfPduQ': "mM/s",
                              'KmPduQPropionaldehyde': "mM",
                              'VmaxrPduQ': "mM/s",
                              'KmPduQPropanol': "mM",

                              'VmaxfPduP': "mM/s",
                              'KmPduPPropionaldehyde': "mM",
                              'VmaxrPduP': "mM/s",
                              'KmPduPPropionylCoA': "mM",

                              'VmaxfPduL': "mM/s",
                              'KmPduLPropionylCoA': "mM",
                              'VmaxrPduL': "mM/s",
                              'KmPduQPropionylPhosphate': "mM",

                              'VmaxfPduW': "mM/s",
                              'KmPduWPropionylPhosphate': "mM",

                              'VmaxfPrpE': "mM/s",
                              'KmPduQPropionate': "mM",

                              'VmaxfPrpC': "mM/s",
                              'KmPduQPropionylCoA': "mM",

                              'nmcps': None,
                              }

VARIABLE_NAMES_UNITS = {'PROPANEDIOL_MCP': "mM",
                        'PROPIONALDEHYDE_MCP': "mM",
                        'PROPANOL_MCP': "mM",
                        'PROPIONYLCOA_MCP': "mM",
                        'PROPIONYLPHOSPHATE_MCP': "mM",

                        'PROPANEDIOL_CYTO': "mM",
                        'PROPIONALDEHYDE_CYTO': "mM",
                        'PROPANOL_CYTO': "mM",
                        'PROPIONYLCOA_CYTO': "mM",
                        'PROPIONYLPHOSPHATE_CYTO': "mM",
                        'PROPIONATE_CYTO': "mM",

                        'PROPANEDIOL_EXT': "mM",
                        'PROPIONALDEHYDE_EXT': "mM",
                        'PROPANOL_EXT': "mM",
                        'PROPIONYLCOA_EXT': "mM",
                        'PROPIONYLPHOSPHATE_EXT': "mM",
                        'PROPIONATE_EXT': "mM"}
