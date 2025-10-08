#!/bin/bash
#SBATCH --account=bdye-delta-gpu
#SBATCH --job-name="test_40_wreg"
#SBATCH --output="/work/hdd/bdye/jxia4/code/autoencoder_mae/logs/test_svo_40.%j.out"
#SBATCH --error="/work/hdd/bdye/jxia4/code/autoencoder_mae/logs/test_svo_40.%j.err"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --mem=100000
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA100x8,gpuH200x8

# Initialize shell environment
source /etc/profile
source ~/.bashrc   # Or other appropriate initialization file

module load anaconda3_gpu
source activate svo-mae

# Load eids from session_order.pkl
session_order_file="/work/hdd/bdye/jxia4/data/tables_and_infos/session_order.pkl"
eids=$(python -c "
import pickle
import numpy as np
session_order = pickle.load(open('$session_order_file', 'rb'))
eids = np.sort(session_order[:40])
#for 160 sessions
#eids = np.delete(eids, [ 12,  58, 123, 139, 146]) #for 160 sessions
print(' '.join(map(str, eids)))
")

export CMD="python -u /work/hdd/bdye/jxia4/code/autoencoder_mae/src/test_perf_GLM_simple_decoder_svoboda_unbalanced_lump.py --eids $eids --with_reg"


srun $CMD

conda deactivate