from utils.utils_profiling import *  # load before other local modules

import argparse
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import dgl
import math
import numpy as np
import torch
import wandb

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Dunbrack_toy import DunbrackDataset

from experiments.dunbrack import models  # as models
from experiments.dunbrack import models_xyz  # as models
import timeit


dir_cache = "cache/"
# m = nn.Sigmoid()  # initialize sigmoid layer
# loss = nn.BCELoss()  # initialize loss function

use_wandb = False

def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def MeanSquaredLoss(yhat, y):
    return torch.mean(torch.sum((yhat-y)**2, -1))

def to_onehot(input, num_classes, toCuda):
    input = input.reshape(len(input), 1)
    res = torch.arange(num_classes).reshape(1, num_classes)
    if toCuda:
        res = res.cuda()
    return (input == res).float()

def to_np(x):
    return x.cpu().detach().numpy()

def train_epoch(epoch, model, loss_fnc, acu_fnc, dataloader, optimizer, scheduler, FLAGS):
    model.train()

    num_iters = len(dataloader)

    racu = 0

    for i, (g, y) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        y = y.to(FLAGS.device)

        optimizer.zero_grad()

        # run model forward and compute loss
        pred = model(g)

        racu += acu_fnc(pred, y)

        loss_ce = loss_fnc(pred, y)
        loss_ce.backward()
        optimizer.step()

        if i % FLAGS.print_interval == 0:
            print(f"[{epoch}|{i}] loss ce: {loss_ce:.5f} [units]")
        if use_wandb:
            if i % FLAGS.log_interval == 0:
                wandb.log({"loss": to_np(loss_ce)})

        if FLAGS.profile and i == 10:
            sys.exit()

        scheduler.step(epoch + i / num_iters)

    racu = racu.type(torch.float64)
    racu /= FLAGS.train_size

    print(f"...[{epoch}|train]: {racu:.5f} [units]")
    if use_wandb:
        wandb.log({"Train accuracy": to_np(racu)})


def val_epoch(epoch, model, loss_fnc, dataloader, FLAGS):
    model.eval()

    rloss = 0
    rloss_l2 = 0
    for i, (g, y) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        y = y.to(FLAGS.device)

        pred = model(g).detach()

        rloss += loss_fnc(pred, y)

    rloss = rloss.type(torch.float64)
    rloss /= FLAGS.val_size

    print(f"...[{epoch}|val]: {rloss:.5f} [units]")

    if use_wandb:
        wandb.log({"Val accuracy": to_np(rloss)})
def test_epoch(epoch, model, acu_fnc, dataloader, FLAGS, dir_cache_epoch):
    model.eval()

    rloss = 0
    for i, (g, y) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        y = y.to(FLAGS.device)

        pred = model(g).detach()

        rloss += acu_fnc(pred, y)

        torch.save(pred, os.path.join(dir_cache_epoch, "pred_" + str(i) + ".pt"))
        torch.save(y, os.path.join(dir_cache_epoch, "y_" + str(i) + ".pt"))

    print("rloss: ", rloss)
    rloss = rloss.type(torch.float64)
    rloss /= FLAGS.test_size
    print(f"...[{epoch}|test]: {rloss:.5f} [units]")
    if use_wandb:
        wandb.log({"Test accuracy": to_np(rloss)})

class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)


def main(FLAGS, UNPARSED_ARGV):
    # Prepare data
    train_dataset = DunbrackDataset(FLAGS.data_address,
                                    FLAGS.task,
                                    FLAGS.dim_output,
                                    mode='train',
                                    embedding=FLAGS.embedding,
                                    coordinate_type=FLAGS.coordinate_type,
                                    # transform=RandomRotation(),
                                    transform=None,
                                    fully_connected=FLAGS.fully_connected)
    train_loader = DataLoader(train_dataset,
                              batch_size=FLAGS.batch_size,
                              shuffle=True,
                              collate_fn=collate,
                              num_workers=FLAGS.num_workers)

    val_dataset = DunbrackDataset(FLAGS.data_address,
                                  FLAGS.task,
                                  FLAGS.dim_output,
                                  mode='valid',
                                  embedding=FLAGS.embedding,
                                  coordinate_type=FLAGS.coordinate_type,
                                  fully_connected=FLAGS.fully_connected)
    val_loader = DataLoader(val_dataset,
                            batch_size=FLAGS.batch_size,
                            shuffle=False,
                            collate_fn=collate,
                            num_workers=FLAGS.num_workers)

    test_dataset = DunbrackDataset(FLAGS.data_address,
                                   FLAGS.task,
                                   FLAGS.dim_output,
                                   mode='test',
                                   embedding=FLAGS.embedding,
                                   coordinate_type=FLAGS.coordinate_type,
                                   fully_connected=FLAGS.fully_connected
                                   )
    test_loader = DataLoader(test_dataset,
                             batch_size=FLAGS.batch_size,
                             shuffle=False,
                             collate_fn=collate,
                             num_workers=FLAGS.num_workers)

    FLAGS.train_size = len(train_dataset)
    FLAGS.val_size = len(val_dataset)
    FLAGS.test_size = len(test_dataset)

    # Choose model
    model = None
    if FLAGS.coordinate_type == "pp":
        model = models.__dict__.get(FLAGS.model)(FLAGS.num_layers,
                                             train_dataset.node_feature_size,
                                             FLAGS.num_channels,
                                             num_nlayers=FLAGS.num_nlayers,
                                             num_degrees=FLAGS.num_degrees,
                                             dim_output=train_dataset.dim_output,
                                             edge_dim=train_dataset.num_edge,
                                             div=FLAGS.div,
                                             pooling=FLAGS.pooling,
                                             n_heads=FLAGS.head)
    elif FLAGS.coordinate_type == "cn":
        model = models_xyz.__dict__.get(FLAGS.model)(FLAGS.num_layers,
                                                     train_dataset.node_feature_size,
                                                     FLAGS.num_channels,
                                                     num_nlayers=FLAGS.num_nlayers,
                                                     num_degrees=FLAGS.num_degrees,
                                                     dim_output=train_dataset.dim_output,
                                                     edge_dim=train_dataset.num_edge,
                                                     div=FLAGS.div,
                                                     pooling=FLAGS.pooling,
                                                     n_heads=FLAGS.head)
    if FLAGS.restore is not None:
        model.load_state_dict(torch.load(FLAGS.restore))
    model.to(FLAGS.device)
    # #wandb.watch(model)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    # optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=0.85)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                               FLAGS.num_epochs,
                                                               eta_min=1e-4)

    # Loss function
    def task_loss_percentage(pred, target):
        pred = pred.cpu()
        target = target.cpu()
        pred = torch.argmax(pred, dim=1)
        return torch.sum(pred == target)

    # Loss function
    def task_loss_cat(pred, target):
        pred = pred.cpu()
        target = target.cpu()
        target = torch.nn.functional.one_hot(target, num_classes=FLAGS.dim_output).float()
        return nn.BCEWithLogitsLoss()(pred, target)

    # Loss function
    def task_loss_coord(pred, target):
        pred = pred.cpu().float()
        target = target.cpu().float()
        # return nn.MSELoss()(pred, target)
        return RMSELoss(pred, target)
        # return MeanSquaredLoss(pred, target)

    def task_auc_coord(pred, target, LEN_ERROR=2.0):
        pred = pred.cpu().float()
        target = target.cpu().float()
        distances = torch.sqrt(torch.sum(torch.square(torch.abs(pred - target)), dim = 1))
        return torch.sum(distances<LEN_ERROR)

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + '.pt')

    # Run training
    print('Begin training')

    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        start_epoch = timeit.default_timer()

        dir_cache_epoch = os.path.join(dir_cache, str(epoch))
        os.mkdir(dir_cache_epoch)

        if FLAGS.task == "target_cat":
            train_epoch(epoch, model, task_loss_cat, task_loss_percentage, train_loader, optimizer, scheduler, FLAGS)
            val_epoch(epoch, model, task_loss_percentage, val_loader, FLAGS)
            test_epoch(epoch, model, task_loss_percentage, test_loader, FLAGS, dir_cache_epoch)
        elif FLAGS.task == "target_coord":
            train_epoch(epoch, model, task_loss_coord, task_auc_coord, train_loader, optimizer, scheduler, FLAGS)
            val_epoch(epoch, model, task_auc_coord, val_loader, FLAGS)
            test_epoch(epoch, model, task_auc_coord, test_loader, FLAGS, dir_cache_epoch)

        stop_epoch = timeit.default_timer()
        print('Time for epoch: ', stop_epoch - start_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model', type=str, default='SE3Transformer',
                        help="String name of model")
    parser.add_argument('--num_layers', type=int, default=4,
                        help="Number of equivariant layers")
    parser.add_argument('--num_degrees', type=int, default=4,
                        help="Number of irreps {0,1,...,num_degrees-1}")
    parser.add_argument('--num_channels', type=int, default=16,
                        help="Number of channels in middle layers")
    parser.add_argument('--num_nlayers', type=int, default=0,
                        help="Number of layers for nonlinearity")
    parser.add_argument('--fully_connected', action='store_true',
                        help="Include global node in graph")
    parser.add_argument('--div', type=float, default=4,
                        help="Low dimensional embedding fraction")
    parser.add_argument('--pooling', type=str, default='avg',
                        help="Choose from avg or max")
    parser.add_argument('--head', type=int, default=1,
                        help="Number of attention heads")

    # Meta-parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=50,
                        help="Number of epochs")

    # Data
    parser.add_argument('--data_address', type=str, default='dunbrack_final.pt',
                        help="Address to structure file")
    # parser.add_argument('--task', type=str, default='homo',
    #         help="QM9 task ['homo, 'mu', 'alpha', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']")
    parser.add_argument('--task', type=str, default='target',
                        help="Dunbrack task ['target_cat' or 'target_coord']")
    parser.add_argument('--dim_output', type=int, default=1,
                        help="Number of categories of Dunbrack task")
    parser.add_argument('--embedding', type=str, default="rota",
                        help="Embedding by rota / eg")
    parser.add_argument('--coordinate_type', type=str, default="pp",
                        help="coordinate type of pp / cn")

    # Logging
    parser.add_argument('--name', type=str, default=None,
                        help="Run name")
    parser.add_argument('--log_interval', type=int, default=25,
                        help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=250,
                        help="Number of steps between printing key stats")
    parser.add_argument('--save_dir', type=str, default="models",
                        help="Directory name to save models")
    parser.add_argument('--restore', type=str, default=None,
                        help="Path to model to restore")
    parser.add_argument('--wandb', type=str, default='equivariant-attention-dunbrack',
                        help="wandb project name")
    parser.add_argument('--use_wandb', action='store_true',
                        help="use wandb project name")

    # Miscellanea
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument('--profile', action='store_true',
                        help="Exit after 10 steps for profiling")

    # Random seed for both Numpy and Pytorch
    parser.add_argument('--seed', type=int, default=None)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()

    # Fix name
    if not FLAGS.name:
        FLAGS.name = f'E-d{FLAGS.num_degrees}-l{FLAGS.num_layers}-{FLAGS.num_channels}-{FLAGS.num_nlayers}'

    # Create model directory
    if not os.path.isdir(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # Fix seed for random numbers
    if not FLAGS.seed: FLAGS.seed = 1992  # np.random.randint(100000)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Automatically choose GPU if available
    FLAGS.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    use_wandb = FLAGS.use_wandb

    if use_wandb:
        # Log all args to wandb
        if FLAGS.name:
            wandb.init(project=f'{FLAGS.wandb}', name=f'{FLAGS.name}')
        else:
            wandb.init(project=f'{FLAGS.wandb}')

    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    # Where the magic is
    start_main = timeit.default_timer()

    if not os.path.isdir(dir_cache):
        os.mkdir(dir_cache)
    dir_cache = os.path.join(dir_cache, str(start_main))
    os.mkdir(dir_cache)

    main(FLAGS, UNPARSED_ARGV)
    stop_main = timeit.default_timer()

    print('Total Time: ', stop_main - start_main)
