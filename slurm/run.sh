#!/bin/bash
# SBATCH --job-name              aptm_tta
# SBATCH --partition             gbunchQ1
# SBATCH --nodes                 1
# SBATCH --tasks-per-node        1
# SBATCH --time                  48:00:00
# SBATCH --mem                   50G
# SBATCH --cpus-per-task         16
# SBATCH --gres                  gpu:1
# SBATCH --output                %j.out
# SBATCH --error                 %j.err

# | Partition Name | Number of GPU in total | GPU | Time limit | Partition QOS | QOS |
# | --- | --- | --- | --- | --- | --- |
# | gbunchQ | 12 | A40 48GB PCIE | 3 days | 4 GPUs pre user simultaneously |  |
# | gbunchQ1 | 3 | 3090 x2 (fstsvr03) V100 x1 (fstsvr02) | 7 days | 4 GPUs pre user simultaneously |  |
# | gbunchQ2 | 12 | A100 80GB PCIE | 2 days | 4 GPUs pre user simultaneously |  |
# | gbunchQ3 | 2 | H800 80GB PCIE | 2 days | 1 GPU pre user simultaneously |  |
#  - Each PI group can occupy maximum 6 GPUs simultaneously - Each user can occupy maximum 4 GPUs simultaneously - Each user can only run 2 jobs, and submit 4 jobs simultaneously

# You can modify the above job parameters
# gres: Number of GPU you want to occupy in this job
# mem: Memory you want to occupy in this job
# time: time limit of this job
# partition: node partition you want to use in this job
# cpus-per-task: cpu cores you want to request in this job

# Insert your commands here
bash /home/user/yc57963/task/APTM/slurm/exp6.0_6.sh
