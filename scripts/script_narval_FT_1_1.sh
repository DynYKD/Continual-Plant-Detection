#!/bin/bash
#SBATCH --time=6:30:00
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
#python -m torch.distributed.launch --nproc_per_node=${GPU} tools/train_first_step.py -c configs/OD_cfg/strawberry_diseases/1-1/e2e_faster_rcnn_R_50_C4_1x.yaml
python tools/trim_detectron_model.py --name "1-1/LR005_BS4_10K"

task=1-1
name="FT_b4_lr0001_2.5K"

for s in {1..6};
do
    python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py \
    -t ${task} -n ${name} -s $s --cls 0 --dist_type none --dataset strawberry_diseases
done