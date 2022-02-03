
#!/bin/bash
#SBATCH --account=b1020
#SBATCH --partition=buyin-dev
#SBATCH --nodes=1
#SBATCH --array=1-4
#SBATCH --ntasks=1
#SBATCH --time=00-00:15:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrearcher2017@u.northwestern.edu
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=true_model
#SBATCH --output=out/true_model_out_%A_%a
#SBATCH --error=err/true_model_err_%A_%a

nsamps=(1e1 1e2)
lambda=(0.01)
beta=(0.01)
#lambda=(0.01 0.1 1)
#beta=(0.01 0.25 0.5)

len_nsamps=${#nsamps[@]}
len_lambda=${#lambda[@]}
len_beta=${#beta[@]}

sublen_lambda_beta=$(( $len_lambda * $len_beta))

zero_index=$(( $SLURM_ARRAY_TASK_ID-1))

module purge all
module load texlive/2020

mpirun -n 1 python main_true_MCMC.py "${nsamps[(((zero_index / $sublen_lambda_beta) % len_nsamps))]}" "adaptive" "${lambda[(((zero_index / $len_beta) % $len_lambda))]}" "${beta[zero_index % $len_beta]}"