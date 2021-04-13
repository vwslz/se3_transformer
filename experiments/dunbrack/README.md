# Dunbrack

Unzip the dataset under dataset directory. This directory is mainly from QM9 directory.

## Experiments
[Log on Wandb](https://wandb.ai/vwslz/equivariant-attention-dunbrack?workspace=user-vwslz)

[Result Comparison](https://docs.google.com/spreadsheets/d/1gb_oLTwC3zbwXrm2OB2Q5SfGFSU6cdRtMmb7RHTa1gg/edit?usp=sharing)

## Training

To train the model, run this command:

for MET-ROTA-5:
```train
python train_dunbrack.py --model SE3Transformer --num_epochs 50 --num_degrees 2 --num_layers 7 --num_channels 32 --num_workers 4 --div 2 --pooling max --head 8  --print_interval 50 --name dunbrack-met --batch_size 16 --data_address 13750_MET-TRIVIAL_ROTA-5_03312021.pt --task target_cat --dim_output 5 --coordinate_type pp --lr 0.001 --use_wandb
```
for TYR-ROTA-2:
```train
python train_dunbrack.py --model SE3Transformer --num_epochs 50 --num_degrees 2 --num_layers 7 --num_channels 32 --num_workers 4 --div 2 --pooling max --head 8 --print_interval 50 --name dunbrack-tyr --batch_size 32 --data_address 44000_TYR-TRIVIAL_ROTA-2_03312021.pt --task target_cat --dim_output 2 --coordinate_type pp --lr 0.001 --use_wandb
```

## Parameters
### Model parameters
1. --model: SE3Transformer
2. --num_layers: Number of equivariant layers
3. --num_degrees: Number of irreps {0,1,...,num_degrees-1}
4. --num_channels: Number of channels in middle layers
5. --num_nlayers: Number of layers for nonlinearity
6. --fully_connected: Include global node in graph
7. --div: Low dimensional embedding fraction
8. --pooling: avg/max
9. --head: Number of attention heads

### Meta-parameters
1. --batch_size: Batch size (16 works on ai institute)
2. --lr: Learning rate 
3. --num_epochs: Number of epochs

### Data
1. --data_address: dunbrack_final.pt
2. --task: target_cat, target_coord
3. --dim_output': varies for dataset. Currently 2.
4. --**embedding**: rota / eg
5. --**coordinate_type**: pp / cn

### Logging
1. --name: Run name
2. --log_interval: Number of steps between logging key stats
3. --print_interval: Number of steps between printing key stats
4. --save_dir: Directory name to save models
5. --restore: Path to model to restore
6. --wandb: Wandb project name
7. --use_wandb: Whether to log on wandb
