import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt
from pdo_model_pytorch import NORM_PRIOR_MEAN_ALL_EXP, NORM_PRIOR_STD_RT_ALL_EXP, N_UNKNOWN_PARAMETERS
from pdo_model_pytorch.pdo_model_log import *
from mpi4py import MPI
import time
from os.path import dirname, abspath
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from scipy.stats import multivariate_normal
from mcmc_sampler import mcmc_sampler
import pickle
ROOT_PATH = dirname(abspath(__file__))

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def main(argv, arc):
	"""
	Generates plots and exp_data_native files of MCMC walks of the posterior distribution of dhaB_dhaT_model

	@param argv[1]: number of MCMC walks
	@param argv[2]: boolean for fixed or adaptive. 0 for fixed, 1 for adaptive
	@param argv[3]: fixed MCMC step size
	@param argv[4]: weighed step size between adaptive MCMC step size and fixed MCMC step size
	@param arc: number of arguments
	@return:
	"""
	seed = int(time.time() * 1e6)
	seed = ((seed & 0xff000000) >> 24) + ((seed & 0x00ff0000) >>  8 ) + ((seed & 0x0000ff00) <<  8 ) + ((seed & 0x000000ff) << 24)
	np.random.seed(seed)
	print("Rank: " + str(rank) + ". The seed is " + str(seed))
	print(argv)
	# get arguments
	nsamps = int(float(argv[1]))
	pdo_model = pdo_model_log()
	sampler = argv[2]

	# preset optional arguments
	lbda = None
	beta = None
	burn_in_n = None
	rprior = lambda n: multivariate_normal.rvs(mean=np.ones(N_UNKNOWN_PARAMETERS),
											   cov=np.diag(np.ones(N_UNKNOWN_PARAMETERS)),
											   size=n)
	# set optional arguments
	if sampler == "fixed":
		lbda = float(argv[3])
		loglik = lambda sample: pdo_model.get_logpost(sample,type = 'qoi only')
		folder_name_data = ROOT_PATH + '/output/MCMC_results_data/' + sampler + "/preset_std/lambda_" \
						   + str(lbda).replace('.', ',') + "_beta_" + str(beta).replace('.', ',')
	elif sampler == "adaptive":
		lbda = float(argv[3])
		beta = float(argv[4])
		loglik = lambda sample: pdo_model.get_logpost(sample, type='qoi only')
		folder_name_data = ROOT_PATH + '/output/MCMC_results_data/' + sampler + "/preset_std/lambda_"\
						   + str(lbda).replace('.', ',') + "_beta_" + str(beta).replace('.', ',')
	elif sampler == "nuts":
		burn_in_n = int(argv[3])
		loglik = lambda sample: pdo_model.get_logpost(sample, type='qoi sens')
		folder_name_data = ROOT_PATH + '/output/MCMC_results_data/' + sampler + "/preset_std/lambda_"\
						   + str(lbda).replace('.', ',') + "_beta_" + str(beta).replace('.', ',')

	time_start = time.time()
	postdraws = mcmc_sampler(sampler, loglik, lambda: rprior(1), nsamp=nsamps, rprior=rprior, lbda=lbda,
							 beta = beta, burn_in_n = burn_in_n)
	time_end = time.time()
	run_time = (time_end-time_start)/float(nsamps)

	print('Run time .................................. : ' + str(np.round(run_time,decimals=3)) + ' seconds per sample')

	# store results
	date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
	folder_name_data += "/nsamples_" + str(nsamps)
	Path(folder_name_data).mkdir(parents=True, exist_ok=True)
	file_name = folder_name_data + "/date_" + date_string + "_rank_" + str(rank) + ".pkl"
	with open(file_name, 'wb') as f:
		pickle.dump(postdraws, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
	main(sys.argv, len(sys.argv))
