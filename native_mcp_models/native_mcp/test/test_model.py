import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt
from native_mcp_models import *
from exp_data_native import TIME_SAMPLES_BOBIK, DATA_SAMPLES, INIT_CONDS, OD_FIT_PARAMS_MCPS_SIGMOID
import time
import sympy as sp
import numpy as np

def test_QoI():
    def sigmoid(x, L, x0, k):
        y = L / (1 + sp.exp(-k * (x - x0)))
        return y
    od_fun = lambda t: sigmoid(t, *OD_FIT_PARAMS_MCPS_SIGMOID)
    params_mean = list(LOG_NORM_MODEL_PRIOR_MEAN.values())
    params_mean = {key: val for key, val in zip(SINGLE_EXP_CALIBRATION_LIST,params_mean)}
    params_mean['VmaxfPduW'] = -np.inf
    params_mean['VmaxfPrpE'] = -np.inf
    params_mean['VmaxfPrpC'] = -np.inf
    params_mean['nmcps'] = 20
    init_conds = {'PROPANEDIOL_MCP_INIT': DATA_SAMPLES['WT'][0,0],
                  'PROPIONALDEHYDE_MCP_INIT': DATA_SAMPLES['WT'][0, 1],
                  'PROPANOL_MCP_INIT': DATA_SAMPLES['WT'][0, 2],
                  'PROPIONYLCOA_MCP_INIT': 0,
                  'PROPIONYLPHOSPHATE_MCP_INIT': 0,

                  'PROPANEDIOL_CYTO_INIT': DATA_SAMPLES['WT'][0,0],
                  'PROPIONALDEHYDE_CYTO_INIT': DATA_SAMPLES['WT'][0, 1],
                  'PROPANOL_CYTO_INIT': DATA_SAMPLES['WT'][0, 2],
                  'PROPIONYLCOA_CYTO_INIT': 0,
                  'PROPIONYLPHOSPHATE_CYTO_INIT': 0,
                  'PROPIONATE_CYTO_INIT': 0,

                  'PROPANEDIOL_EXT_INIT': DATA_SAMPLES['WT'][0,0],
                  'PROPIONALDEHYDE_EXT_INIT': DATA_SAMPLES['WT'][0, 1],
                  'PROPANOL_EXT_INIT': DATA_SAMPLES['WT'][0, 2],
                  'PROPIONYLCOA_EXT_INIT': 0,
                  'PROPIONYLPHOSPHATE_EXT_INIT': 0,
                  'PROPIONATE_EXT_INIT': DATA_SAMPLES['WT'][0, 3]}
    params = {**params_mean, **init_conds}

    pdu_mcp_model = pdu_mcp_model_log(MCP_VOLUME_SPHERICAL, MCP_SURFACE_AREA_SPHERICAL, od_fun)
    FLAG, qoi_vals, _ = pdu_mcp_model.get_sol_sens(params, evaluation_times=np.linspace(0,TIME_SAMPLES_BOBIK[-1] + 10),
                                                       type= 'qoi sens',rtol=10**-1)
    plt.plot(np.linspace(0,TIME_SAMPLES_BOBIK[-1] + 10),qoi_vals[:,0])
    plt.scatter(TIME_SAMPLES_BOBIK,DATA_SAMPLES['WT'][:,0])
    plt.title('Plot of external 1,2-PD')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    plt.plot(np.linspace(0,TIME_SAMPLES_BOBIK[-1] + 10),qoi_vals[:,1])
    plt.scatter(TIME_SAMPLES_BOBIK,DATA_SAMPLES['WT'][:,1])
    plt.title('Plot of external propionaldehyde')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    plt.plot(np.linspace(0,TIME_SAMPLES_BOBIK[-1] + 10),qoi_vals[:,2])
    plt.scatter(TIME_SAMPLES_BOBIK,DATA_SAMPLES['WT'][:,2])

    plt.title('Plot of external propanol')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

    plt.plot(np.linspace(0,TIME_SAMPLES_BOBIK[-1] + 10),qoi_vals[:,3])
    plt.scatter(TIME_SAMPLES_BOBIK,DATA_SAMPLES['WT'][:,3])
    plt.title('Plot of external propionate')
    plt.xlabel('time (hr)')
    plt.ylabel('concentration (mM)')
    plt.show()

# def test_logpost():
#     sample = np.array([-0.7428849, 0.28148294, 0.42938577, -0.12950951, -1.54237829, 1.0974601, 2.23714432, -2.3220673,
#               -1.07838844, -0.9317926, -4.90275089, -0.29460825, 2.63943847, 0, 0, 0, 0,0,0, 0,0,0, 0,0,0])
#     # sample = np.zeros_like(NORM_PRIOR_MEAN_ALL_EXP)
#     # sample[0] = (-4- LOG_NORM_MODEL_PRIOR_MEAN['PermCellGlycerol'])/LOG_NORM_MODEL_PRIOR_STD['PermCellGlycerol'] # [2.5]
#     # sample[1] = (-5- LOG_NORM_MODEL_PRIOR_MEAN['PermCellPDO'])/LOG_NORM_MODEL_PRIOR_STD['PermCellPDO'] # [2.5]
#     # sample[2] = (-4 - LOG_NORM_MODEL_PRIOR_MEAN['PermCell3HPA'])/LOG_NORM_MODEL_PRIOR_STD['PermCell3HPA'] # [2.5]
#     # sample[3] = (2.5- LOG_NORM_MODEL_PRIOR_MEAN['VmaxfDhaB'])/LOG_NORM_MODEL_PRIOR_STD['VmaxfDhaB'] # [2.5]
#     # sample[4] = (-1 - LOG_NORM_MODEL_PRIOR_MEAN['KmDhaBG'])/LOG_NORM_MODEL_PRIOR_STD['KmDhaBG']  # [-1]
#     # sample[5] = (3 - LOG_NORM_MODEL_PRIOR_MEAN['VmaxfDhaT'])/LOG_NORM_MODEL_PRIOR_STD['VmaxfDhaT']  # [3]
#     # sample[6] = (-2 - LOG_NORM_MODEL_PRIOR_MEAN['KmDhaTH'])/LOG_NORM_MODEL_PRIOR_STD['KmDhaTH']  # [-1]
#     # sample[7] = (1.5 - LOG_NORM_MODEL_PRIOR_MEAN['VmaxfMetab'])/LOG_NORM_MODEL_PRIOR_STD['VmaxfMetab'] # [1.5]
#     # sample[6] = (-1 - LOG_NORM_MODEL_PRIOR_MEAN['KmMetabG'])/LOG_NORM_MODEL_PRIOR_STD['KmMetabG']  # [-1]
#
#     # time_start = time.time()
#     # print(logpost(sample, type = 'qoi only', rtol=10**-1))
#     # time_end = time.time()
#     # print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#     #
#     # time_start = time.time()
#     # print(logpost(sample, type = 'qoi only', rtol=10**-2))
#     # time_end = time.time()
#     # print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#     #
#     time_start = time.time()
#     print(logpost(sample, type = 'qoi only', rtol=10**-3))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     print(logpost(sample,type = 'qoi only',rtol=10**-4))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     print(logpost(sample,rtol=10**-1))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     print(logpost(sample,rtol=10**-2))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     print(logpost(sample,rtol=10**-3))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     print(logpost(sample,rtol=10**-4))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
# def test_logpost_atol():
#     sample = np.zeros_like(NORM_PRIOR_MEAN_ALL_EXP)
#
#     time_start = time.time()
#     rtol = 10 ** -2
#     atol = 10 ** -7
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -2
#     atol = 10 ** -8
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -2
#     atol = 10 ** -9
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -3
#     atol = 10 ** -7
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -3
#     atol = 10 ** -8
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#
#     time_start = time.time()
#     rtol = 10 ** -3
#     atol = 10 ** -9
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -4
#     atol = 10 ** -5
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -4
#     atol = 10 ** -4
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -4
#     atol = 10 ** -3
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -4
#     atol = 10 ** -2
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -5
#     atol = 10 ** -5
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -5
#     atol = 10 ** -4
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -5
#     atol = 10 ** -3
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -5
#     atol = 10 ** -2
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol=atol, type="qoi only")))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start) / 60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -1
#     atol = 10 ** -7
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol = atol)))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -1
#     atol = 10 ** -8
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " +  str(logpost(sample, rtol=rtol, atol = atol)))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -2
#     atol = 10 ** -5
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " +  str(logpost(sample, rtol=rtol, atol = atol)))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -2
#     atol = 10 ** -4
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol = atol)))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -2
#     atol = 10 ** -3
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol = atol)))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -2
#     atol = 10 ** -2
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol = atol)))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -2
#     atol = 10 ** -1
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " + str(logpost(sample, rtol=rtol, atol = atol)))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -3
#     atol = 10 ** -5
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " +  str(logpost(sample, rtol=rtol, atol = atol)))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -3
#     atol = 10 ** -4
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " +  str(logpost(sample, rtol=rtol, atol = atol)))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -3
#     atol = 10 ** -3
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " +  str(logpost(sample, rtol=rtol, atol = atol)))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -3
#     atol = 10 ** -2
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " +  str(logpost(sample, rtol=rtol, atol = atol)))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')
#
#     time_start = time.time()
#     rtol = 10 ** -3
#     atol = 10 ** -1
#     print("rtol = " + str(rtol) + ", atol = " + str(atol) + ": " +  str(logpost(sample, rtol=rtol, atol = atol)))
#     time_end = time.time()
#     print('This run took ' + str((time_end - time_start)/60) + ' minutes')

if __name__ == '__main__':
    test_QoI()
    # test_logpost()
    # test_logpost_atol()
