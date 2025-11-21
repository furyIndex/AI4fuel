
import os, glob, json
import numpy as np
import pandas as pd
from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

def _safe_index(x, arr, default=0):
    try:
        return arr.index(x)
    except ValueError:
        return default

def smiles_to_data(smi: str, y: float = None, row_id: int = None) -> Data:

    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError(f"Invalid SMILES: {smi}")


    types, chirs = [], []
    for a in m.GetAtoms():
        types.append(_safe_index(a.GetAtomicNum(), ATOM_LIST, default=len(ATOM_LIST)-1))
        chirs.append(_safe_index(a.GetChiralTag(), CHIRALITY_LIST, default=0))
    x = torch.stack([torch.tensor(types, dtype=torch.long),
                     torch.tensor(chirs, dtype=torch.long)], dim=1)  # (N,2)

    # edges (two directed)
    row, col, eattr = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = _safe_index(b.GetBondType(), BOND_LIST, default=0)
        bd = _safe_index(b.GetBondDir(), BONDDIR_LIST, default=0)
        row += [i, j]; col += [j, i]
        eattr += [[bt, bd], [bt, bd]]

    if not row:
        row, col = [0], [0]
        eattr = [[_safe_index(BT.SINGLE, BOND_LIST, 0), 0]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr  = torch.tensor(np.asarray(eattr), dtype=torch.long)

    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        d.y = torch.tensor([float(y)], dtype=torch.float32)
    if row_id is not None:
        d.row_id = int(row_id)
    return d


def _find_fold_paths(root, sheet):
    for pat in [os.path.join(root, sheet, "fold_*", "split.json"),
                os.path.join(root, f"sheet_{sheet}", "fold_*", "split.json")]:
        ps = sorted(glob.glob(pat))
        if ps: return ps
    return []

def _pick_smiles_and_label_columns(df: pd.DataFrame, smiles_hint: str = "SMILES") -> Tuple[str, str]:
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError("列数不足：至少需要 SMILES 与 label 两列")
    label_col = cols[-2]  # 倒数第二列
    smiles_col = None
    for c in cols:
        if c.lower() == smiles_hint.lower():
            smiles_col = c; break
    if smiles_col is None:
        for c in cols:
            if c.lower() == "smiles":
                smiles_col = c; break
    if smiles_col is None:
        smiles_col = cols[-1]
    if smiles_col == label_col:
        smiles_col, label_col = cols[-1], cols[-2]
    return smiles_col, label_col


class _ExcelSplitTorchDataset(Dataset):
    def __init__(self, smiles: List[str], y: np.ndarray, indices: List[int]):
        self.items = []
        for idx in indices:
            smi = smiles[idx]
            m = Chem.MolFromSmiles(smi)
            if m is None:
                raise ValueError(f"Row {idx}: invalid SMILES")
            smi = Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
            self.items.append(smiles_to_data(smi, y[idx], row_id=idx))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

class MolExcelFinetuneDatasetWrapper(object):
    def __init__(self, batch_size: int, **cfg):
        self.batch_size = batch_size
        self.num_workers = int(cfg.get("num_workers", 4))
        self.xlsx = cfg["xlsx"]
        self.splits_root = cfg["splits_root"]
        self.sheet = cfg["sheet"]
        self.smiles_col_hint = cfg.get("smiles_col", "SMILES")
        self.fold_id = cfg.get("fold_id", None)

        df = pd.read_excel(self.xlsx, sheet_name=self.sheet)
        self.smiles_col, self.label_col = _pick_smiles_and_label_columns(df, self.smiles_col_hint)
        self.smiles = df[self.smiles_col].astype(str).tolist()
        self.y = pd.to_numeric(df[self.label_col], errors="coerce").values.astype(np.float32)

        fps = _find_fold_paths(self.splits_root, self.sheet)
        if not fps:
            raise FileNotFoundError(f"[{self.sheet}] 未找到 split.json：{self.splits_root}/(sheet_){self.sheet}/fold_x/split.json")
        if self.fold_id:
            cand = [p for p in fps if os.path.basename(os.path.dirname(p)) == self.fold_id]
            self.spath = cand[0] if cand else fps[0]
        else:
            self.spath = fps[0]

        with open(self.spath, "r", encoding="utf-8") as f:
            sp = json.load(f)
        self.tr_idx = sp["inner_train_idx"]; self.va_idx = sp["inner_val_idx"]; self.te_idx = sp["outer_test_idx"]
        self.fold_id = os.path.basename(os.path.dirname(self.spath))

    def get_data_loaders(self):
        ds_tr = _ExcelSplitTorchDataset(self.smiles, self.y, self.tr_idx)
        ds_va = _ExcelSplitTorchDataset(self.smiles, self.y, self.va_idx)
        ds_te = _ExcelSplitTorchDataset(self.smiles, self.y, self.te_idx)
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True,  num_workers=self.num_workers)
        dl_va = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        dl_te = DataLoader(ds_te, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dl_tr, dl_va, dl_te
