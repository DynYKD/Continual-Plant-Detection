#!/bin/bash
#SBATCH --time=18:00:00
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
python -m torch.distributed.launch --nproc_per_node=${GPU} tools/train_first_step.py -c configs/OD_cfg/OPPD/26-20/e2e_faster_rcnn_R_50_C4_1x.yaml
python tools/trim_detectron_model.py --name "26-20/LR005_BS4_50K"

task=26-20
name="FT_b4_lr001_50K"

s=1
python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py -t ${task} -n ${name} -s $s --cls 0 --dist_type none