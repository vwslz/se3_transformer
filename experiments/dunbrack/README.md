# Dunbrack

Unzip the dataset under dataset directory

## Training

To train the model, run this command:

```train
python train_dunbrack.py --model SE3Transformer --num_epochs 50 --num_degrees 4 --num_layers 7 --num_channels 32 --name dunbrack-chi --num_workers 4 --batch_size 16 --task chi --num_cat_task 2 --div 2 --pooling max --head 8 --lr 0.001 --print_interval 50 --data_address dunbrack_final.pt --use_wandb
```
