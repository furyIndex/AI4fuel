/data/home/Longh/.virtualenvs/ai4fuel/bin/python train_moleculeNet.py \
    --csv ../data/molnet_desc/Lipophilicity_desc.csv \
    --dataset Lipophilicity \
    --mapping ../descriptors_group/descriptorsMap/descriptorsMapping.json \
    --exp_k 10 \
    --seed 42 \
    --device cuda:0 \
    --trials 50 \
    --epochs_embed 1000 \
    --patience_embed 100