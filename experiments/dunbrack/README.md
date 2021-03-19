# Dunbrack

Unzip the dataset under dataset directory. This directory is mainly from QM9 directory.

## Experiments
[Log on Wandb](https://wandb.ai/vwslz/equivariant-attention-dunbrack?workspace=user-vwslz)

## Training

To train the model, run this command:

```train
python train_dunbrack.py --model SE3Transformer --num_epochs 50 --num_degrees 4 --num_layers 7 --num_channels 32 --name dunbrack-chi --num_workers 4 --batch_size 16 --task target --num_cat_task 2 --div 2 --pooling max --head 8 --lr 0.001 --print_interval 50 --data_address dunbrack.pt --use_wandb
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
2. --task: chi
3. --num_cat_task': varies for dataset. Currently 2.

### Logging
1. --name: Run name
2. --log_interval: Number of steps between logging key stats
3. --print_interval: Number of steps between printing key stats
4. --save_dir: Directory name to save models
5. --restore: Path to model to restore
6. --wandb: Wandb project name
7. --use_wandb: Whether to log on wandb
