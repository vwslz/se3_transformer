
![experiments](https://github.com/vwslz/se3_transformer/blob/master/data/dl-experiments.png)
* Backbone indepedent rotamer coordinate files were collected from: http://kinemage.biochem.duke.edu/databases/rotkins.php
* Rotamers for each residue were clustered at 2A based on their chi2 `terminal atom` (e.g.,):
 
   ``` # 1 Rotamer for ALA, GLY, PRO
    TERMINAL_ATOM_MAP = {
    'ARG' : 'CD', # 5 Rotamers
    'ASN' : 'OD1', # 4 Rotamers
    'ASP' : 'OD1', # 3 Rotamers
    'CYS' : 'SG', # 2 Rotamers
    'GLN' : 'CD', # 4 Rotamers
    'GLU' : 'CD', # 5 Rotamers
    'HIS' : 'ND1', # 5 Rotamers
    'ILE' : 'CD1', # 4 Rotamers
    'LEU' : 'CD1', # 3 Rotamers
    'LYS' : 'CD', # 5 Rotamers
    'MET' : 'SD', # 5 Rotamers
    'PHE' : 'CD1', # 2 Rotamers
    'SER' : 'OG', # 2 Rotamers
    'THR' : 'OG1', # 2 Rotamers
    'TRP' : 'CD1', # 6 Rotamers
    'TYR' : 'CD1', # 2 Rotamers
    'VAL' : 'CG1' # 2 Rotamers
    }
  ``` 
 * A [Dunbrack dataset](http://dunbrack.fccc.edu/Guoli/pisces_download.php) was used at the onset.  Specifically, the high resolution, 80% sequence-similar [dataset](http://dunbrack.fccc.edu/Guoli/culledpdb_hh/cullpdb_pc80_res2.0_R0.25_d210225_chains22717.gz)
* Rotamers for each residue in the dataset were categorized based on their `terminal_atom` 
  - Here's a histogram showing the rmsd of the `terminal_atom`:
* Node features include:
  - ca_coords (x,y,z)
  - res_type (one_hot representation)
  - chi_category (int)
   - source node is obfuscated
  - phi/psi (deg) or c_coords/n_coords (vector based on ca_coords)
* Edge information for experiments don't contain any edge_features (e.g., `[source_node, neighbor_node, edge_feature=0]`)
