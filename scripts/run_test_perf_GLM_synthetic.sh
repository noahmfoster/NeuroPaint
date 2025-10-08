#!/bin/bash
#SBATCH --account=bdye-delta-gpu
#SBATCH --job-name="te_syn_reconreg"
#SBATCH --output="/work/hdd/bdye/jxia4/code/autoencoder_mae/logs/test_syn.%j.out"
#SBATCH --error="/work/hdd/bdye/jxia4/code/autoencoder_mae/logs/test_syn.%j.err"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=100000
#SBATCH --partition=gpuA100x4-interactive,gpuA40x4-interactive,gpuA100x8-interactive,gpuH200x8-interactive

# Initialize shell environment
source /etc/profile
source ~/.bashrc   # Or other appropriate initialization file

module load anaconda3_gpu
source activate svo-mae


export CMD="python -u /work/hdd/bdye/jxia4/code/autoencoder_mae/src/test_perf_GLM_simple_decoder.py"


srun $CMD

conda deactivate