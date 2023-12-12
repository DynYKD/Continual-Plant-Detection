#!/bin/bash
#SBATCH --time=3:00:00
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

alias exp="python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/test_same_backbone.py"
shopt -s expand_aliases

# FIRST STEP
#### 10-2
task=36-10
cls=1
name="same_backbone_mask${loss_mask}_cls${cls}_b4_lr001"
s=1
exp -t ${task} -n ${name} --rpn --uce --dist_type uce --cls ${cls} -s $s --dataset OPPD
