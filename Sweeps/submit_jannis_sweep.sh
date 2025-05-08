#!/bin/sh
#BSUB -q gpua100
#BSUB -J jannis
#BSUB -W 4:00
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB s183995@student.dtu.dk

module load cuda/12.4
module load gcc/13.3.0-binutils-2.42

source "/zhome/22/c/137477/miniforge3/etc/profile.d/conda.sh"
conda activate s183995
nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048
python train_tabular_sweep.py --sweep_id udo92xy5 --data_dir /work3/s183995/Tabular/data/processed --dataset_name jannis --wandb_project Tabular --wandb_entity tensorGang --task classification --tt_method ridge_cholesky --tt_track_eval