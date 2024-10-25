#!/bin/bash
#SBATCH --job-name=rl-dt
#SBATCH --output=./data/rl-dt.txt
#SBATCH -p stampede
#SBATCH -N 1
#SBATCH -n 1

source ~/.bashrc
conda activate rl-powergrid

python ./scripts/decision_transformer/generate.py baseline 0 1000
python ./scripts/decision_transformer/generate.py alpha 0 1000
python ./scripts/decision_transformer/generate.py alpha 1 1000
python ./scripts/decision_transformer/generate.py alpha 2 1000
python ./scripts/decision_transformer/generate.py bravo 0 1000
python ./scripts/decision_transformer/generate.py bravo 1 1000
python ./scripts/decision_transformer/generate.py bravo 2 1000

python ./scripts/decision_transformer/train.py baseline 0
python ./scripts/decision_transformer/train.py alpha 0
python ./scripts/decision_transformer/train.py alpha 1
python ./scripts/decision_transformer/train.py alpha 2
python ./scripts/decision_transformer/train.py bravo 0
python ./scripts/decision_transformer/train.py bravo 1
python ./scripts/decision_transformer/train.py bravo 2
