# -*- coding: utf-8 -*-
"""
Drop-in 替换版 getFeatures.py：
- 显式使用 RDKit canonical=True, isomericSmiles=True 与训练脚本保持一致；
- 保持“0 键分子”支持（bond 特征 0×D，占位遮罩）。
"""
from functools import partial
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import pickle, time

# 如果你的 Featurizer 在此处：保持不变
from other_baseline.attentiveFP.afp_model.Featurizer import atom_features, bond_features

degrees = [0, 1, 2, 3, 4, 5]

class MolGraph(object):
    def __init__(self):
        self.nodes = {}
    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node
    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))
    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)
        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)
        self.nodes[ntype] = new_nodes
    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])
    def rdkit_ix_array(self): return np.array([node.rdkit_ix for node in self.nodes['atom']])
    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor] for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]

class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix
    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)
    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]

class memoize(object):
    def __init__(self, func):
        self.func = func; self.cache = {}
    def __call__(self, *args):
        if args in self.cache: return self.cache[args]
        result = self.func(*args); self.cache[args] = result; return result
    def __get__(self, obj, objtype): return partial(self.__call__, obj)

def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node
    # bonds（可能为0条）
    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))
    if 'bond' not in graph.nodes:
        graph.nodes['bond'] = []
    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph

def array_rep_from_smiles(molgraph):
    atom_feat = molgraph.feature_array('atom')
    # 0 键：bond 特征用 0×D 的二维数组占位
    if 'bond' in molgraph.nodes and len(molgraph.nodes['bond']) > 0:
        bond_feat = molgraph.feature_array('bond')
    else:
        # 用一个极简分子 'CC' 推断 bond 维度
        D = len(bond_features(Chem.MolFromSmiles('CC').GetBonds()[0]))
        bond_feat = np.zeros((0, D), dtype=float)
        molgraph.nodes.setdefault('bond', [])
    arrayrep = {
        'atom_features': atom_feat,
        'bond_features': bond_feat,
        'atom_list'    : molgraph.neighbor_list('molecule', 'atom'),
        'rdkit_ix'     : molgraph.rdkit_ix_array()
    }
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep

def gen_descriptor_data(smilesList):
    smiles_to_fingerprint_array = {}
    for i,smiles in enumerate(smilesList):
        # 与训练脚本一致：canonical + isomeric
        m = Chem.MolFromSmiles(smiles)
        smiles_cano = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) if m is not None else None
        if smiles_cano is None:
            continue
        try:
            molgraph = graph_from_smiles(smiles_cano)
            molgraph.sort_nodes_by_degree('atom')
            arrayrep = array_rep_from_smiles(molgraph)
            smiles_to_fingerprint_array[smiles_cano] = arrayrep
        except Exception:
            print(smiles_cano); time.sleep(0.05)
    return smiles_to_fingerprint_array

def get_smiles_dicts(smilesList):
    # 与 save_smiles_dicts 相同，只是不写文件
    return _build_feature_dicts(smilesList)

def save_smiles_dicts(smilesList, filename):
    feature_dicts = _build_feature_dicts(smilesList)
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(feature_dicts, f)
    print('feature dicts file saved as ' + filename + '.pickle')
    return feature_dicts

def _build_feature_dicts(smilesList):
    max_atom_len = 0; max_bond_len = 0
    num_atom_features = 0; num_bond_features_ = 0
    smiles_to_rdkit_list = {}
    smiles_to_fingerprint_features = gen_descriptor_data(smilesList)

    for smiles, arrayrep in smiles_to_fingerprint_features.items():
        atom_features_arr = arrayrep['atom_features']; bond_features_arr = arrayrep['bond_features']
        rdkit_list = arrayrep['rdkit_ix']; smiles_to_rdkit_list[smiles] = rdkit_list
        atom_len, num_atom_features = atom_features_arr.shape
        bond_len, num_bond_features_ = bond_features_arr.shape
        max_atom_len = max(max_atom_len, atom_len)
        max_bond_len = max(max_bond_len, bond_len)

    max_atom_index_num = max_atom_len; max_bond_index_num = max_bond_len
    max_atom_len += 1; max_bond_len += 1

    smiles_to_atom_info = {}; smiles_to_bond_info = {}
    smiles_to_atom_neighbors = {}; smiles_to_bond_neighbors = {}
    smiles_to_atom_mask = {}

    for smiles, arrayrep in smiles_to_fingerprint_features.items():
        mask = np.zeros((max_atom_len))
        atoms = np.zeros((max_atom_len, num_atom_features))
        bonds = np.zeros((max_bond_len, num_bond_features_))
        atom_neighbors = np.zeros((max_atom_len, len(degrees)))
        bond_neighbors = np.zeros((max_atom_len, len(degrees)))
        atom_neighbors.fill(max_atom_index_num); bond_neighbors.fill(max_bond_index_num)

        atom_features_arr = arrayrep['atom_features']; bond_features_arr = arrayrep['bond_features']

        for i, feature in enumerate(atom_features_arr):
            mask[i] = 1.0; atoms[i] = feature
        for j, feature in enumerate(bond_features_arr):
            bonds[j] = feature

        atom_neighbor_count = 0; bond_neighbor_count = 0
        for degree in degrees:
            atom_neighbors_list = arrayrep[('atom_neighbors', degree)]
            bond_neighbors_list = arrayrep[('bond_neighbors', degree)]
            if len(atom_neighbors_list) > 0:
                for degree_array in atom_neighbors_list:
                    for j, value in enumerate(degree_array):
                        atom_neighbors[atom_neighbor_count, j] = value
                    atom_neighbor_count += 1
            if len(bond_neighbors_list) > 0:
                for degree_array in bond_neighbors_list:
                    for j, value in enumerate(degree_array):
                        bond_neighbors[bond_neighbor_count, j] = value
                    bond_neighbor_count += 1

        smiles_to_atom_info[smiles] = atoms
        smiles_to_bond_info[smiles] = bonds
        smiles_to_atom_neighbors[smiles] = atom_neighbors
        smiles_to_bond_neighbors[smiles] = bond_neighbors
        smiles_to_atom_mask[smiles] = mask

    del smiles_to_fingerprint_features
    return {
        'smiles_to_atom_mask': smiles_to_atom_mask,
        'smiles_to_atom_info': smiles_to_atom_info,
        'smiles_to_bond_info': smiles_to_bond_info,
        'smiles_to_atom_neighbors': smiles_to_atom_neighbors,
        'smiles_to_bond_neighbors': smiles_to_bond_neighbors,
        'smiles_to_rdkit_list': smiles_to_rdkit_list
    }

def get_smiles_array(smilesList, feature_dicts):
    x_mask = []; x_atom = []; x_bonds = []; x_atom_index = []; x_bond_index = []
    for smiles in smilesList:
        x_mask.append(feature_dicts['smiles_to_atom_mask'][smiles])
        x_atom.append(feature_dicts['smiles_to_atom_info'][smiles])
        x_bonds.append(feature_dicts['smiles_to_bond_info'][smiles])
        x_atom_index.append(feature_dicts['smiles_to_atom_neighbors'][smiles])
        x_bond_index.append(feature_dicts['smiles_to_bond_neighbors'][smiles])
    return (np.asarray(x_atom), np.asarray(x_bonds),
            np.asarray(x_atom_index), np.asarray(x_bond_index),
            np.asarray(x_mask), feature_dicts['smiles_to_rdkit_list'])
