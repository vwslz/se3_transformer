import os
import json
from glob import glob

import numpy as np
import prody as pdy
import pandas as pd

JSON_DIR = './out_redo_w_rot-cat_v2'
CLEAN_PDB_DIR = './pdb_clean'

def read_json(json_file):
    with open(json_file, 'r') as rf:
        return json.load(rf)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_pdb(json_file):
    pdb_name = os.path.basename(json_file).split('.')[0]
    return os.path.abspath(f'{CLEAN_PDB_DIR}/{pdb_name}.pdb')

RES_ID = []
RES_NAME = []
PHI = []
PSI = []
CHI = []
NUM_NODE = []
NUM_EDGE = []
X = [] # CA COORDS
Xn = [] # N COORDS
Xc = [] # C COORDS
TARGET = [] # TARGET CATEGORY
EDGE = []
MAX_EDGES = 49
MAX_NODES = 50
RES_CATEGORIES = np.array([109, 110, 111]) # Rotamer categories for Tyrosine
RES_TYPES = np.array(
    ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
     'SER', 'THR', 'TRP', 'TYR', 'VAL'], dtype='str')

jsons = glob(f'.{JSON_DIR}/*.json')
pdb_search_cutoff = 15000
pdb_environment = []

for _idx, res_cat in enumerate(np.tile(RES_CATEGORIES, 5000)):
    # Searching for source nodes
    _json = jsons[_idx]
    json_data = read_json(_json)
    protein = pdy.parsePDB(get_pdb(_json))
    protein = protein.select('heavy')
    df = pd.DataFrame.from_dict(json_data)
    _res_categories = df['rota_category'].values

    flag = True
    if res_cat not in _res_categories:

        while flag:
            _json = jsons[pdb_search_cutoff]
            json_data = read_json(_json)
            protein = pdy.parsePDB(get_pdb(_json))
            df = pd.DataFrame.from_dict(json_data)
            _res_cat = df['rota_category'].values

            pdb_search_cutoff += 1
            if res_cat in _res_cat:
                flag = False

    # Get the feature info based on selected source node
    nodes = df[df['rota_category'] == res_cat]['res_num'].values
    _pdb_name = os.path.basename(_json).split('.')[0]
    _phi = df['phi'].values
    _psi = df['psi'].values
    _ca_xyz = df['ca_xyz'].values
    _c_xyz = df['c_xyz'].values
    _n_xyz = df['n_xyz'].values
    _o_xyz = df['o_xyz'].values
    _res_name = df['res_name'].values
    _chis = df['chis'].values
    _res_cat = df['rota_category'].values

    for source_node in nodes[:1]:
        # node features
        data = df[(df['res_num'] == int(source_node))]
        res_name = data['res_name'].values[0]
        res_num = data['res_num'].values[0]
        phi = data['phi'].values[0]
        psi = data['psi'].values[0]
        chis = data['chis'].values[0]
        target_chi_category = abs(108 - data['rota_category'].values[0]) # shift the target category to index-1
        res_id = _pdb_name + '_' + str(source_node)
        RES_ID.append(res_id)
        TARGET.append(target_chi_category)

        edge = []
        d_mat = np.array(data['res_d_mat'])
        d_mat = np.array(d_mat[0][0])
        neighbor_1_indicies = [neighbor for neighbor in list(np.where(d_mat < 10.0)[0]) if neighbor != data.index[0]]

        for dst in neighbor_1_indicies:
            edge.append([data.index[0], dst, 0])

        # reformatting edge info
        list_node = []
        for e in edge:
            list_node.append(e[0])
            list_node.append(e[1])
        list_node = list(set(list_node))
        neighbor_dict = dict(zip(list_node, [i for i in range(len(list_node))]))
        re_indexed_edge = []
        for e in edge:
            e[0] = neighbor_dict[e[0]]
            e[1] = neighbor_dict[e[1]]
        num_edge = len(edge)
        num_nodes = len(list_node)

        # ensuring that all edge info have same dimension across all neighborhoods
        norm_edge = np.zeros((MAX_EDGES, 3))
        for i, v in enumerate(edge):
            norm_edge[i] = v
        EDGE.append([norm_edge])
        NUM_EDGE.append([num_edge])
        NUM_NODE.append([num_nodes])

        # ensuring that all node features have same dimension across all neighborhoods
        norm_x = np.zeros((MAX_NODES, 3))
        norm_xc = np.zeros((MAX_NODES, 3))
        norm_xn = np.zeros((MAX_NODES, 3))
        norm_phi = np.zeros((MAX_NODES, 1))
        norm_psi = np.zeros((MAX_NODES, 1))
        norm_resname = np.zeros((MAX_NODES, 20), dtype=bool)
        norm_chis = np.zeros((MAX_NODES, 1))

        for idx, i in enumerate(list_node):
            i = i.astype(int)
            norm_x[idx] = [_ca_xyz[i][0], _ca_xyz[i][1], _ca_xyz[i][2]]
            norm_xc[idx] = [_c_xyz[i][0], _c_xyz[i][1], _c_xyz[i][2]]
            norm_xn[idx] = [_n_xyz[i][0], _n_xyz[i][1], _n_xyz[i][2]]
            norm_phi[idx] = _phi[i]
            norm_psi[idx] = _psi[i]
            t = RES_TYPES == _res_name[i]
            norm_resname[idx] = list(t)
            norm_chis[idx] = int(_res_cat[i])
            if i == data.index[0]:
                norm_chis[idx] = 0

        PHI.append(norm_phi)
        PSI.append(norm_psi)
        RES_NAME.append(norm_resname)    
        X.append(norm_x)
        Xc.append(norm_xc)
        Xn.append(norm_xn)
        CHI.append(norm_chis)

data_d = {
    'res_id': RES_ID,
    'num_node': NUM_NODE,
    'num_edge': NUM_EDGE,
    'target': TARGET,
    'chis': CHI,  # category
    'x': X,
    'x_c': Xc,
    'x_n': Xn,
    'one_hot': RES_NAME,
    'phi': PHI,
    'psi': PSI,
    'edge': EDGE
}

data_dunbrack = {}

data_dunbrack["train"] = {}
data_dunbrack["valid"] = {}
data_dunbrack["test"] = {}

# edit it here
split_train_valid = 920
split_valid_test = 1035

for key in data_d.keys():
    data_dunbrack["train"][key] = data_d[key][0:split_train_valid]
    data_dunbrack["valid"][key] = data_d[key][split_train_valid:split_valid_test]
    data_dunbrack["test"][key] = data_d[key][split_valid_test:]

torch.save(data_dunbrack, './bb_indp-equiv_res_type-single-node.pt')