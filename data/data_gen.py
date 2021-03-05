import os
import json
from glob import glob

import numpy as np
import prody as pdy
import pandas as pd

jsons = glob('./out_redo_w_rot-cat_v2/*.json')


def read_json(json_file):
    with open(json_file, 'r') as rf:
        return json.load(rf)

def get_pdb(json_file):
    pdb_name = os.path.basename(json_file).split('.')[0]
    return os.path.abspath(f'./pdb_clean/{pdb_name}.pdb')


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


def get_separated_nodes(df):

    df_Y = df[(df['rota_category'] == 109) | (df['rota_category'] == 110) | (df['rota_category'] == 111)]
    Y = list(df_Y.index)
    if len(Y) > 1:
        largest_group = 0
        best_group = None
        for idx, i in enumerate(Y):
            d_mat = np.array(df.iloc[[i]]['res_d_mat'])[0][0]
            group = [neighbor for neighbor in list(np.where(np.array(d_mat) > 20.0)[0]) if neighbor in Y]
            if len(group) > largest_group:
                largest_group = len(group)
                best_group = group
                source_node = i
        
        if best_group:
            largest_subgroup = 0
            for idx, j in enumerate(best_group):
                d_mat = np.array(df.iloc[[j]]['res_d_mat'])[0][0]
                subgroup = [neighbor for neighbor in list(np.where(np.array(d_mat) > 20.0)[0]) if neighbor in Y]
                if len(subgroup) > largest_subgroup:
                    largest_subgroup = len(subgroup)
                    best_subgroup = subgroup
                    best_idx = idx
                    sub_source_node = j
            try:
                seperated_nodes = [source_node, sub_source_node, np.intersect1d(best_group, best_subgroup).tolist()[0]]
            except:
                seperated_nodes = [source_node]
        else:
            seperated_nodes = [Y[0]]
    else:
        try:
            seperated_nodes = [Y[0]]        
        except:
            seperated_nodes = None

    return seperated_nodes

MAX_EDGES = 49
MAX_NODES = 50
cutoff = 20000
RES_TYPES = np.array(['TYR'], dtype='str')
pdb_environment = []
for _idx, res in enumerate(np.tile(RES_TYPES, 20000)):
    _json = jsons[_idx]
    json_data = read_json(_json)
    protein = pdy.parsePDB(get_pdb(_json))
    protein = protein.select('heavy')
    df = pd.DataFrame.from_dict(json_data)
    _res_categories = df['rota_category'].values
    _res_name = df['res_name'].values

    flag = True
    if res not in _res_name:

        while flag:
            _json = jsons[cutoff]
            json_data = read_json(_json)
            protein = pdy.parsePDB(get_pdb(_json))
            df = pd.DataFrame.from_dict(json_data)
            _res_name = df['res_name'].values

            cutoff = cutoff + 1
            if res in _res_name:
                flag = False
    
    nodes = get_separated_nodes(df)
    
    if nodes:

        _pdb_name = os.path.basename(_json).split('.')[0]
        _phi = df['phi'].values
        _psi = df['psi'].values
        _ca_xyz = df['ca_xyz'].values
        _c_xyz = df['c_xyz'].values
        _n_xyz = df['n_xyz'].values
        _o_xyz = df['o_xyz'].values
        _res_name = df['res_name'].values
        _chis = df['chis'].values
        _chi_category = df['rota_category'].values


        for source_node in nodes:
            data = df.iloc[[source_node]]
            res_name = data['res_name'].values[0]
            res_num = data['res_num'].values[0]
            phi = data['phi'].values[0]
            psi = data['psi'].values[0]
            chis = data['chis'].values[0]
            target_chi_category = abs(108 - data['rota_category'].values[0])
            res_id = _pdb_name + '_' + str(res_num)
            RES_ID.append([res_id])

            TARGET.append([target_chi_category])

            edge = []
            d_mat = np.array(data['res_d_mat'])
            d_mat = np.array(d_mat[0][0])
            neighbor_1_indicies = [neighbor for neighbor in list(np.where(d_mat < 10.0)[0]) if neighbor != data.index[0]]

            for dst in neighbor_1_indicies:
                edge.append([data.index[0], dst, 0])

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

            norm_edge = np.zeros((MAX_EDGES, 3))
            for i, v in enumerate(edge):
                norm_edge[i] = v
            EDGE.append([norm_edge])
            NUM_EDGE.append([num_edge])
            NUM_NODE.append([num_nodes])

            # getting coord_info info
            norm_x = np.zeros((MAX_NODES, 3))
            norm_xc = np.zeros((MAX_NODES, 3))
            norm_xn = np.zeros((MAX_NODES, 3))
            norm_phi = np.zeros((MAX_NODES, 1))
            norm_psi = np.zeros((MAX_NODES, 1))
            norm_resname = np.zeros((MAX_NODES, 20), dtype=bool)
            norm_chis = np.zeros((MAX_NODES, 1))

            REAL_RES_TYPES = np.array(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'], dtype='str')

            for idx, i in enumerate(list_node):
                i = i.astype(int)
                norm_x[idx] = [_ca_xyz[i][0], _ca_xyz[i][1], _ca_xyz[i][2]]
                norm_xc[idx] = [_c_xyz[i][0], _c_xyz[i][1], _c_xyz[i][2]]
                norm_xn[idx] = [_n_xyz[i][0], _n_xyz[i][1], _n_xyz[i][2]]
                norm_phi[idx] =_phi[i]
                norm_psi[idx] = _psi[i]
                t = REAL_RES_TYPES == _res_name[i]
                norm_resname[idx] = list(t)
                norm_chis[idx] = int(_chi_category[i])
                if i == data.index[0]:
                    norm_chis[idx] = 0
                    
            data_d = {
                'res_id' : res_id,
                'num_node' : num_nodes,
                'num_edge' : num_edge,
                'target' : target_chi_category,
                'chis' : norm_chis, # category
                'x' : norm_x,
                'x_c' : norm_xc,
                'x_n' : norm_xn,
                'one_hot' : norm_resname,
                'phi' : norm_phi,
                'psi' : norm_psi,
                'edge' : norm_edge
            }
            
            pdb_environment.append(data_d)

with open('./TYROSINE_ALL_NODES.json', 'a') as fp:
    json.dump(pdb_environment, fp, cls=NpEncoder, indent=4)
fp.close()
