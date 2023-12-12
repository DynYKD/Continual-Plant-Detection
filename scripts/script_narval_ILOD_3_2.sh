#!/bin/bash
#SBATCH --time=1:30:00
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

task=3-2
name="ILOD_b4_lr0001_5K"

for s in {1..2};
do
    python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py -t ${task} -n ${name} -s $s --dataset strawberry_diseases
done