#!/bin/bash
#SBATCH --time=3:30:00
#SBATCH --account=def-chaibdra
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1

source ~/cisal/bin/activate
module load python/3.7 cuda cudnn
cd ~/projects/def-chaibdra/mapaf2/agriculture_vision_2023/MMA
#!/bin/bash

port=$(python get_free_port.py)
GPU=1

task=26-20
s=1
cls=1
name="FILOD_b4_lr001_10K"
python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py -t ${task} -n ${name} --feat std --rpn -s $s