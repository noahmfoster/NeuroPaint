#!/bin/bash
#SBATCH --account=bdye-delta-gpu
#SBATCH --job-name="test_20ibl"
#SBATCH --output="/work/hdd/bdye/jxia4/code/autoencoder_mae/logs/test_ibl20.%j.out"
#SBATCH --error="/work/hdd/bdye/jxia4/code/autoencoder_mae/logs/test_ibl20.%j.err"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --mem=100000
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA100x8,gpuH200x8

# Initialize shell environment
source /etc/profile
source ~/.bashrc   # Or other appropriate initialization file

module load anaconda3_gpu
source activate svo-mae


#load session id for ibl
session_order_file="/work/hdd/bdye/jxia4/data/tables_and_infos/ibl_eids.txt"
eids=$(python -c "with open('$session_order_file', 'r') as file: print('\n'.join([line.strip() for line in file]))")


# Print loaded eids for debugging
echo "Loaded eids: $eids"

export CMD="python -u /work/hdd/bdye/jxia4/code/autoencoder_mae/src/test_perf_ibl.py --eids $eids --with_reg"


srun $CMD

conda deactivate