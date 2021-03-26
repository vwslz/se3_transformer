Originally cloned from [SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks](https://github.com/FabianFuchsML/se3-transformer-public).

# Installation
(recommended commands with conda)
- [pytorch](https://pytorch.org/)
- [dgl](https://www.dgl.ai/)
- pip install packaging
- conda install -c conda-forge pynfft
- pip install lie_learn
- pip install wandb
- **pip install -e .**

# Experiment
[experiments/dunbrack/](https://github.com/vwslz/se3_transformer/tree/master/experiments/dunbrack)

# Dataset
[se3_transformer/data/](https://github.com/vwslz/se3_transformer/tree/master/data)

# Results:
![image](https://user-images.githubusercontent.com/46386583/112658086-6f79f900-8e29-11eb-80b9-c2303caf98c5.png)
using `/home/vwslz/github/workspace/se3_transformer/experiments/dunbrack/train_dunbrack.py --model SE3Transformer --num_epochs 50 --num_degrees 4 --num_layers 7 --num_channels 32 --name dunbrack-chi --num_workers 4 --batch_size 16 --task target --num_cat_task 2 --div 2 --pooling max --head 8 --lr 0.00075 --print_interval 50 --data_address dunbrack.pt --use_wandb`
