import os
import prody as pdy
import pandas as pd

# data from http://dunbrack.fccc.edu/Guoli/pisces_download.php
csv = './misc/cullpdb_pc90_res2.5_R1.0_d210225_chains36493.csv' # only PDB-CHAIN field


def fetch_pdb(data=csv):
    df = pd.read_csv(data, sep='\t')
    PDB = list(set([i[:4] for i in df['IDs'].values]))
    fetched = pdy.fetchPDB(PDB)
    return fetched


def clean_pdb(pdb_file, chain, write_out=True):
    pdb = pdy.parsePDB(pdb_file)
    p = pdb.select(f'stdaa and chain {chain}')

    # Ensure that all resnums are above 0
    res_nums = [res_num for res_num in sorted(list(set(p.getResnums()))) if res_num > 0]

    # Remove Insertion Codes
    clean_res_num = []
    for res_num in res_nums:
        if len(set(p.select(f'resnum {res_num}').getIcodes())) > 1:
            clean_res_num.append(f'{res_num}_')  # "_" selects residue with no insertion codes
        else:
            clean_res_num.append(str(res_num))

    clean_protein = p.select('resnum {}'.format(' '.join(clean_res_num)))

    pdb_name = os.path.basename(pdb_file).split('.')[0]

    if write_out:
        pdy.writePDB(f'./pdb_clean_new/{pdb_name}_{chain}.pdb', clean_protein)

    # Select cleaned residues
    return clean_protein
