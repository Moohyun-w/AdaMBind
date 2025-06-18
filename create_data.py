#!/usr/bin/python
import pandas as pd
import numpy as np
import os
from rdkit import Chem
import networkx as nx
from utils import *
from utils.TestbedDataset import TestbedDataset

unconvert=[]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    try:
        mol = Chem.MolFromSmiles(smile)  
        c_size = mol.GetNumAtoms()  
    
        features = []
        for atom in mol.GetAtoms(): 
            feature = atom_features(atom)
            features.append( feature / sum(feature) )  

       
        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])
            
        return c_size, features, edge_index
    except Exception as e:
        unconvert.append(smile)

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000


r='C:/Users/l/Desktop/MalTaDTA/data'
datasets = ['davis'] 
compound_iso_smiles = []
for dt_name in datasets: 
    opts = ['full']
    for opt in opts:
        df = pd.read_csv(f"{r}/{dt_name}-{opt}-data.csv")
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g


for dataset in datasets:
    processed_data_file_train = f"{r}//processed/{dataset}-full-data.pt'"
    # processed_data_file_test = r+'processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train))):
        df = pd.read_csv(f"{r}/{dt_name}-{opt}-data.csv")
        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
        XT = [seq_cat(t) for t in train_prots]  
        train_drug, train_prot,  train_y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
        print('preparing ', dataset + '.pt in pytorch format!')
        train_data = TestbedDataset(root=f"{r}/", dataset=f'{dataset}-full-data', xd=train_drug, xt=train_prot, y=train_y,smile_graph=smile_graph)
    
    else:
        print(processed_data_file_train, ' and ', processed_data_file_train, ' are already created')        
