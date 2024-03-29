'''
The DhaB-DhaT model contains DhaB-DhaT reaction pathway
in the MCP; diffusion in the cell; diffusion from the cell 
in the external volume.

This model is currently in use. The DhaB-DhaT model assumes that there 
are N identical cells within the external volume.

Programme written by aarcher07
Editing History:
- 28/10/20
'''


import sympy as sp
from .model_constants import *
from .prior_constants import *
import signal

class TimeOutException(Exception):
    pass

def handler(signum, frame):
    raise TimeOutException("Integration is taking too long")

signal.signal(signal.SIGALRM, handler)

class pdu_mcp_model:

    def __init__(self, mcp_volume, mcp_surface_area, cell_conc_od,
                 cell_volume = CELL_VOLUME, cell_surface_area = CELL_SURFACE_AREA,
                 external_volume = EXTERNAL_VOLUME):
        """

        :param mcp_volume: volume of MCP
        :param mcp_surface_area: surface area of MCP
        :param cell_conc_od: OD of cell concentration as a continuous, differentiable, closed form function
        :param cell_volume: cell volume
        :param cell_surface_area: cell surface area
        :param external_volume: external volume
        """
        # geometric parameters
        self.mcp_volume = mcp_volume
        self.mcp_surface_area = mcp_surface_area
        self.cell_volume = cell_volume
        self.cell_surface_area = cell_surface_area
        self.external_volume = external_volume

        #ratios
        self.MCP_surf_MCP_vol_ratio = mcp_surface_area / mcp_volume
        self.cell_surf_cell_vol_ratio = cell_surface_area / cell_volume
        self.MCP_surf_cell_vol_ratio = mcp_surface_area / cell_volume
        self.cell_surf_external_vol_ratio = cell_surface_area / external_volume

        #integration parameters
        self.nvars = N_VARIABLES
        self.nparams = len(SINGLE_EXP_CALIBRATION_LIST)
        self.cell_conc_od = lambda t: cell_conc_od(t)

    def _RHS(self, t, x, params):
        """
        Computes the spatial derivative of the system at time point, t
        :param t: time
        :param x: state variables
        :param params: parameter list
        """
        ###################################################################################
        ################################# Initialization ##################################
        ###################################################################################
        params_dict = {param_name: param_val for param_name, param_val in zip(SINGLE_EXP_CALIBRATION_LIST,params)}
        # Integration Parameters
        assert len(x) == self.nvars
        # differential equation parameters
        d = np.zeros((len(x))).tolist()  # convert to list to allow use of symbolic derivatives

        # cell growth
        # differential equation parameters
        ncells = self.cell_conc_od(t / HRS_TO_SECS) * OD_TO_CELL_COUNT
        nmcps = params_dict['nmcps']

        ###################################################################################
        ################################## MCP reactions ##################################
        ###################################################################################
        R_PduCDE = params_dict["VmaxfPduCDE"] * x[0] / (x[0] + params_dict["KmPduCDEPropanediol"])

        R_PduQf = params_dict["VmaxfPduQ"] * x[1] / (x[1] + params_dict["KmPduQPropionaldehyde"])
        R_PduQr = params_dict["VmaxrPduQ"] * x[2] / (x[2] + params_dict["KmPduQPropanol"])

        R_PduPf = params_dict["VmaxfPduP"] * x[1] / (x[1] + params_dict["KmPduPPropionaldehyde"])
        R_PduPr = params_dict["VmaxrPduP"] * x[3] / (x[3] + params_dict["KmPduPPropionylCoA"])

        R_PduLf = params_dict["VmaxfPduL"] * x[3] / (x[3] + params_dict["KmPduLPropionylCoA"])
        R_PduLr = params_dict["VmaxrPduL"] * x[4] / (x[4] + params_dict["KmPduQPropionylPhosphate"])

        d[0] = -R_PduCDE + self.MCP_surf_MCP_vol_ratio * params_dict['PermMCPPropanediol'] * (
                    x[5] - x[0])  # microcompartment equation for Propanediol
        d[1] =  -R_PduPf + R_PduPr - R_PduQf + R_PduQr + R_PduCDE + self.MCP_surf_MCP_vol_ratio * params_dict['PermMCPPropionaldehyde'] * (
                    x[6] - x[1])  # microcompartment equation for Propionaldehyde
        d[2] = R_PduQf - R_PduQr + self.MCP_surf_MCP_vol_ratio * params_dict['PermMCPPropanol'] * (
                    x[7] - x[2])  # microcompartment equation for Propanol
        d[3] = R_PduPf - R_PduPr - R_PduLf + R_PduLr + self.MCP_surf_MCP_vol_ratio * params_dict['PermMCPPropionylCoA'] * (
                    x[8] - x[3])  # microcompartment equation for PropionylCoA'
        d[4] = R_PduLf - R_PduLr  + self.MCP_surf_MCP_vol_ratio * params_dict['PermMCPPropionylPhosphate'] * (
                    x[9] - x[4])  # microcompartment equation for PropionylPhosphate

        ####################################################################################
        ##################################### cytosol of cell ##############################
        ####################################################################################

        R_PduWf = params_dict["VmaxfPduW"] * x[9] / (x[9] + params_dict["KmPduWPropionylPhosphate"])

        R_PrpEf = params_dict["VmaxfPrpE"] * x[10] / (x[10] + params_dict["KmPrpEPropionate"])

        R_PrpCf = params_dict["VmaxfPrpC"] * x[8] / (x[8] + params_dict["KmPrpCPropionylCoA"])

        d[5] = - params_dict['PermCellPropanediol'] * self.cell_surf_cell_vol_ratio * (x[5] - x[11]) \
               - nmcps * params_dict['PermMCPPropanediol'] * self.MCP_surf_cell_vol_ratio * (x[5] - x[0])

        d[6] = - params_dict['PermCellPropionaldehyde'] * self.cell_surf_cell_vol_ratio * (x[6] - x[12]) \
               - nmcps * params_dict['PermMCPPropionaldehyde'] * self.MCP_surf_cell_vol_ratio * (x[6] - x[1])

        d[7] = - params_dict['PermCellPropanol'] * self.cell_surf_cell_vol_ratio * (x[7] - x[13]) \
               - nmcps * params_dict['PermMCPPropanol'] * self.MCP_surf_cell_vol_ratio * (x[7] - x[2])

        d[8] = -R_PrpCf - params_dict['PermCellPropionylCoA'] * self.cell_surf_cell_vol_ratio * (x[8] - x[14]) \
               - nmcps * params_dict['PermMCPPropionylCoA'] * self.MCP_surf_cell_vol_ratio * (x[8] - x[3])

        d[9] = -R_PduWf - params_dict['PermCellPropionylPhosphate'] * self.cell_surf_cell_vol_ratio * (x[9] - x[15]) - \
               nmcps * params_dict['PermMCPPropionylPhosphate'] * self.MCP_surf_cell_vol_ratio * (x[9] - x[4])

        d[10] = R_PduWf - R_PrpEf - params_dict['PermCellPropionate'] * self.cell_surf_cell_vol_ratio * (x[10] - x[16])

        #####################################################################################
        ######################### external volume equations #################################
        #####################################################################################

        d[11] = ncells * self.cell_surf_external_vol_ratio * params_dict['PermCellPropanediol'] * (x[5] - x[11])
        d[12] = ncells * self.cell_surf_external_vol_ratio * params_dict['PermCellPropionaldehyde'] * (x[6] - x[12])
        d[13] = ncells * self.cell_surf_external_vol_ratio * params_dict['PermCellPropanol'] * (x[7] - x[13])
        d[14] = ncells * self.cell_surf_external_vol_ratio * params_dict['PermCellPropionylCoA'] * (x[8] - x[14])
        d[15] = ncells * self.cell_surf_external_vol_ratio * params_dict['PermCellPropionylPhosphate'] * (x[9] - x[15])
        d[16] = ncells * self.cell_surf_external_vol_ratio * params_dict['PermCellPropionate'] * (x[10] - x[16])

        return d

