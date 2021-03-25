import os
import sys

import dgl
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from scipy.constants import physical_constants

# hartree2eV = physical_constants['hartree-electron volt relationship'][0]
DTYPE = np.float32
DTYPE_INT = np.int32
DTYPE_LONG = np.long

class DunbrackDataset(Dataset):
    """Dunbrack dataset."""
    num_edge = 1 # num of edges
    node_feature_size = 23 #
    num_chi = 2 # add in dataset
    input_keys = [
                  'res_id',
                  'phi',
                  'psi',
                  'num_node',
                  'num_edge',
                  'target',
                  'x',
                  'x_c',
                  'x_n',
                  'one_hot',
                  'chis',
                  'edge'
                  ]

    unit_conversion = {'chi': 1.0}

    def __init__(self, file_address: str, task: str, num_cat_task: int, mode: str = 'train',
                 transform=None, fully_connected: bool = False):
        """Create a dataset object

        Args:
            file_address: path to data
            task: target task ["chis"]
            mode: [train/val/test] mode
            transform: data augmentation functions
            fully_connected: return a fully connected graph
        """
        self.file_address = file_address
        self.task = task
        self.mode = mode
        self.transform = transform
        self.fully_connected = fully_connected
        self.num_cat_task = num_cat_task  # add in dataset

        # Encode and extra bond type for fully connected graphs
        self.num_edge += fully_connected

        self.load_data()
        self.len = len(self.targets)
        print(f"Loaded {mode}-set, task: {task}, source: {self.file_address}, length: {len(self)}")

    def __len__(self):
        return self.len

    def load_data(self):
        # Load dict and select train/valid/test split
        data = torch.load(self.file_address)
        data = data[self.mode]

        # Filter out the inputs
        self.inputs = {key: np.array(data[key]) for key in self.input_keys}

        # Filter out the targets and population stats
        self.targets = data[self.task]

        # TODO: use the training stats unlike the other papers
        self.mean = np.mean(self.targets)
        self.std = np.std(self.targets)

    def get_target(self, idx, normalize=True):
        target = self.targets[idx]
        if normalize:
            target = (target - self.mean) / self.std
        return target

    def norm2units(self, x, denormalize=True, center=True):
        # Convert from normalized to representation
        if denormalize:
            x = x * self.std
            # Add the mean: not necessary for error computations
            if not center:
                x += self.mean
        x = self.unit_conversion[self.task] * x
        return x

    def to_one_hot(self, data, num_classes):
        one_hot = np.zeros(list(data.shape) + [num_classes])
        one_hot[np.arange(len(data)), data] = 1
        return one_hot

    # def _get_adjacency(self, n_atoms):
    #     # Adjust adjacency structure
    #     seq = np.arange(n_atoms)
    #     src = seq[:, None] * np.ones((1, n_atoms), dtype=np.int32)
    #     dst = src.T
    #     ## Remove diagonals and reshape
    #     src[seq, seq] = -1
    #     dst[seq, seq] = -1
    #     src, dst = src.reshape(-1), dst.reshape(-1)
    #     src, dst = src[src > -1], dst[dst > -1]
    #
    #     return src, dst

    def get(self, key, idx):
        return self.inputs[key][idx]

    def connect_fully(self, edges, num_node):
        """Convert to a fully connected graph"""
        # Initialize all edges: no self-edges
        adjacency = {}
        for i in range(num_node):
            for j in range(num_node):
                if i != j:
                    # adjacency[(i, j)] = self.num_edge - 1
                    adjacency[(i, j)] = 0

        # Given edges to be given weights, currently set to 0 for all
        # # Add bonded edges
        for idx in range(edges.shape[0]):
            adjacency[(edges[idx, 0], edges[idx, 1])] = edges[idx, 2]
            adjacency[(edges[idx, 1], edges[idx, 0])] = edges[idx, 2]

        # Convert to numpy arrays
        src = []
        dst = []
        w = []
        for edge, weight in adjacency.items():
            src.append(edge[0])
            dst.append(edge[1])
            w.append(weight)

        return np.array(src), np.array(dst), np.array(w)

    def connect_partially(self, edge):
        src = np.concatenate([edge[:, 0], edge[:, 1]])
        dst = np.concatenate([edge[:, 1], edge[:, 0]])
        w = np.concatenate([edge[:, 2], edge[:, 2]])
        return src, dst, w

    def __getitem__(self, idx):
        # Load node features
        num_node = self.get('num_node', idx)
        x = self.get('x', idx)[:num_node].astype(DTYPE)
        # x_c = self.get('x_c', idx)[:num_node].astype(DTYPE)
        # x_n = self.get('x_n', idx)[:num_node].astype(DTYPE)
        one_hot = self.get('one_hot', idx)[:num_node].astype(DTYPE)
        phi = self.get('phi', idx)[:num_node].astype(DTYPE)
        psi = self.get('psi', idx)[:num_node].astype(DTYPE)
        chis = self.get('chis', idx)[:num_node].astype(DTYPE)
        # chis_new = []
        # for i_chis in range(chis.shape[0]):
        #     chis_new.append(self.to_one_hot(chis[i_chis], self.num_chi)[0])
        # chis_new = np.array(chis_new).astype(DTYPE)

        # Load edge features
        num_edge = self.get('num_edge', idx)
        edge = self.get('edge', idx)[:num_edge]
        edge = np.asarray(edge, dtype=DTYPE_INT)

        # Load target
        y = self.get('target', idx).astype(DTYPE_LONG)
        # y = self.get_target(idx, normalize=True).astype(DTYPE)
        # y = np.array([y])
        # y = self.to_one_hot(y, self.num_chi).astype(DTYPE_INT)

        # Augmentation on the coordinates
        if self.transform:
            x = self.transform(x).astype(DTYPE)

        # Create nodes
        if self.fully_connected:
            src, dst, w = self.connect_fully(edge, num_node)
        else:
            src, dst, w = self.connect_partially(edge)
        w = self.to_one_hot(w, self.num_edge).astype(DTYPE)

        # Create graph
        G = dgl.DGLGraph((src, dst))

        # Add node features to graph
        G.ndata['x'] = torch.tensor(x)  # [num_node,3]
        G.ndata['f'] = torch.tensor(np.concatenate([phi, psi, one_hot, chis], -1)[..., None])  # [num_node,23,1]
        # G.ndata['f'] = torch.tensor(np.concatenate([d_c, d_n, one_hot, chis], -1)[..., None])  # [num_node,27,1]

        # Add edge features to graph
        G.edata['d'] = torch.tensor(x[dst] - x[src])
        G.edata['w'] = torch.tensor(w)
        return G, y


if __name__ == "__main__":
    def collate(samples):
        graphs, y = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(y)


    train_dataset = DunbrackDataset('./dunbrack.pt',
                               "chis",
                               mode='train',
                               transform=None)
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              collate_fn=collate,
                              num_workers=4)

    for data in train_loader:
        print("MINIBATCH")
        print(data)
        sys.exit()


