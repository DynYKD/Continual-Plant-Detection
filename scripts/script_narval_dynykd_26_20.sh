#!/bin/bash
#SBATCH --time=4:00:00
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

#First step
#python -m torch.distributed.launch --nproc_per_node=${GPU} tools/train_first_step.py -c configs/OD_cfg/6-1/e2e_mask_rcnn_R_50_C4_1x.yaml
#python tools/trim_detectron_model.py --name "6-1/LR005_BS4_10K"

task=26-20

s=1
cls=1
name="dynykd_b4_lr001_rpn_10K_cls${cls}"
python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental_same_backbone.py -t ${task} -n ${name} --rpn -s $s