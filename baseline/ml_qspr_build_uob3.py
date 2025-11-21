import json
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem


def is_aromatic_carbon(a: rdchem.Atom) -> bool:
    return a.GetAtomicNum()==6 and a.GetIsAromatic()

def is_sp3_carbon(a: rdchem.Atom) -> bool:
    return a.GetAtomicNum()==6 and a.GetHybridization()==rdchem.HybridizationType.SP3 and (not a.GetIsAromatic())

def carbon_neighbors(a: rdchem.Atom) -> int:
    return sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum()==6)

def is_methyl(a: rdchem.Atom) -> bool:
    return a.GetAtomicNum()==6 and a.GetTotalNumHs()==3 and carbon_neighbors(a)==1 and (not a.GetIsAromatic())

def is_methylene(a: rdchem.Atom) -> bool:
    return a.GetAtomicNum()==6 and a.GetTotalNumHs()==2 and (not a.GetIsAromatic())

def is_ring_methylene(a: rdchem.Atom) -> bool:
    return is_methylene(a) and a.IsInRing()

def is_nonring_methylene(a: rdchem.Atom) -> bool:
    return is_methylene(a) and (not a.IsInRing())

def is_tertiary_carbon(a: rdchem.Atom) -> bool:
    return a.GetAtomicNum()==6 and carbon_neighbors(a)==3 and a.GetTotalNumHs()>=1 and (not a.GetIsAromatic())

def is_quaternary_carbon(a: rdchem.Atom) -> bool:
    return a.GetAtomicNum()==6 and carbon_neighbors(a)==4 and a.GetTotalNumHs()==0 and (not a.GetIsAromatic())

def count_smarts(mol, smarts: str) -> int:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None: return 0
    return len(mol.GetSubstructMatches(patt))

def aromatic_CC_bond_count(mol) -> int:
    return sum(1 for b in mol.GetBonds()
               if b.GetIsAromatic()
               and b.GetBeginAtom().GetAtomicNum()==6
               and b.GetEndAtom().GetAtomicNum()==6)

def max_linear_run(mol, atom_filter) -> int:
    nodes = [a.GetIdx() for a in mol.GetAtoms() if atom_filter(a)]
    node_set, visited = set(nodes), set()
    best = 0
    for start in nodes:
        if start in visited: continue
        stack=[start]; comp=0
        while stack:
            i=stack.pop()
            if i in visited: continue
            visited.add(i); comp+=1
            ai = mol.GetAtomWithIdx(i)
            for nb in ai.GetNeighbors():
                j = nb.GetIdx()
                if j in node_set:
                    stack.append(j)
        best = max(best, comp)
    return best


def _fused_ring_systems(mol):
    ri = mol.GetRingInfo()
    bond_rings = ri.BondRings()
    n = len(bond_rings)
    parent = list(range(n))

    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra!=rb: parent[rb]=ra


    bond2rings = {}
    for i, r in enumerate(bond_rings):
        for b in r:
            bond2rings.setdefault(b, []).append(i)
    for b, rings in bond2rings.items():
        for i in range(1, len(rings)):
            union(rings[0], rings[i])
    comp = {}
    for i in range(n):
        root = find(i)
        comp.setdefault(root, []).append(i)

    bond2comp = {}
    for root, rs in comp.items():
        for r in rs:
            for b in bond_rings[r]:
                bond2comp[b] = root

    atom2comp = {a: set() for a in range(mol.GetNumAtoms())}
    for b_idx, root in bond2comp.items():
        b = mol.GetBondWithIdx(b_idx)
        atom2comp[b.GetBeginAtomIdx()].add(root)
        atom2comp[b.GetEndAtomIdx()].add(root)
    return comp, atom2comp, bond2comp, bond_rings

def _aromatic_attachment_class(mol, arom_atom_idx):

    from collections import Counter, deque
    comp, atom2comp, bond2comp, bond_rings = _fused_ring_systems(mol)
    comps = list(atom2comp[arom_atom_idx])
    if not comps: return 0, 99
    cid = comps[0]
    ring_ids = comp[cid]
    ring_count = len(ring_ids)


    cnt = Counter()
    for r in ring_ids:
        for b in bond_rings[r]:
            cnt[b] += 1
    fusion_bonds = {b for b, v in cnt.items() if v >= 2 and mol.GetBondWithIdx(b).GetIsAromatic()}
    if not fusion_bonds:
        return ring_count, 99

    vis = set([arom_atom_idx])
    dq = deque([(arom_atom_idx, 0)])
    while dq:
        u, d = dq.popleft()
        au = mol.GetAtomWithIdx(u)
        for nb in au.GetNeighbors():
            b = mol.GetBondBetweenAtoms(u, nb.GetIdx())
            if b.GetIsAromatic() and b.GetIdx() in fusion_bonds:
                return ring_count, d
        for nb in au.GetNeighbors():
            b = mol.GetBondBetweenAtoms(u, nb.GetIdx())
            if not b.GetIsAromatic():
                continue
            v = nb.GetIdx()
            if v not in vis:
                vis.add(v); dq.append((v, d+1))
    return ring_count, 99

def _bucket_ab(ring_count, dist_to_fusion):

    if ring_count <= 1:
        return 0                    # AB1: benzene-like
    if ring_count == 2:
        return 1 if dist_to_fusion <= 1 else 2   # AB2 (alpha), AB3 (beta)
    if ring_count == 3:
        return 3 if dist_to_fusion <= 1 else 4   # AB4 (near fusion), AB5 (peripheral)
    if dist_to_fusion <= 1: return 5   # AB6
    if dist_to_fusion == 2: return 6   # AB7
    if dist_to_fusion == 3: return 7   # AB8
    if dist_to_fusion == 4: return 8   # AB9
    if dist_to_fusion == 5: return 9   # AB10
    if dist_to_fusion == 6: return 10  # AB11
    if dist_to_fusion == 7: return 11  # AB12
    if dist_to_fusion == 8: return 12  # AB13
    return 13                          # AB14

def enumerate_aromatic_attachments(mol):

    out=[]
    for b in mol.GetBonds():
        a1,a2=b.GetBeginAtom(), b.GetEndAtom()
        if b.GetBondTypeAsDouble() < 1.0:
            continue
        for ringC, aliphC in ((a1,a2),(a2,a1)):
            if is_aromatic_carbon(ringC) and is_sp3_carbon(aliphC):
                c_nei = carbon_neighbors(aliphC) - 1
                out.append((ringC.GetIdx(), aliphC.GetIdx(), c_nei>=2))
                break
    return out

def load_ab_smarts(json_path=None):
    if not json_path: return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    compiled={}
    for k, pats in data.items():
        compiled[k] = [Chem.MolFromSmarts(p) for p in pats if Chem.MolFromSmarts(p) is not None]
    return compiled

def ab14_counts(mol, ab_smarts_compiled=None):

    atts = enumerate_aromatic_attachments(mol)
    total = len(atts)
    branched_bins = [0]*14
    unbranched_sum = 0

    if ab_smarts_compiled:
        for arom_idx, _, is_branched in atts:
            if is_branched:
                assigned=False
                for i in range(14):
                    key=f"{1}.{i+1}"
                    if key in ab_smarts_compiled:
                        for patt in ab_smarts_compiled[key]:
                            for match in mol.GetSubstructMatches(patt):
                                if arom_idx in match:
                                    branched_bins[i]+=1; assigned=True; break
                            if assigned: break
                    if assigned: break
            else:
                unbranched_sum += 1
        return branched_bins + [unbranched_sum, total]


    for arom_idx, _, is_branched in atts:
        if not is_branched:
            unbranched_sum += 1
            continue
        ring_count, d = _aromatic_attachment_class(mol, arom_idx)
        idx = _bucket_ab(ring_count, d)
        branched_bins[idx] += 1

    return branched_bins + [unbranched_sum, total]


def groups_2_to_27(mol):
    g2  = aromatic_CC_bond_count(mol)
    g3  = count_smarts(mol, "[C;R]=[C;R]")
    g4  = count_smarts(mol, "[C;!R]=[C;!R]")
    g5  = count_smarts(mol, "[C]#[C]")
    g6  = sum(1 for a in mol.GetAtoms() if is_tertiary_carbon(a))
    g7  = sum(1 for a in mol.GetAtoms() if is_quaternary_carbon(a))
    g8  = sum(1 for a in mol.GetAtoms() if is_methyl(a))
    g9  = max_linear_run(mol, is_nonring_methylene)
    g10 = sum(1 for a in mol.GetAtoms() if is_nonring_methylene(a))
    g11 = max_linear_run(mol, is_ring_methylene)
    g12 = sum(1 for a in mol.GetAtoms() if is_ring_methylene(a))
    g13 = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==6 and a.GetTotalNumHs()==1 and (not is_tertiary_carbon(a)))
    g14 = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==6 and a.GetTotalNumHs()==0 and (not is_quaternary_carbon(a)))
    g15 = count_smarts(mol, "[OX2H]")
    g16 = count_smarts(mol, "[OD2;!R]([#6])[#6]")
    g17 = count_smarts(mol, "[OD2;R]([#6])[#6]")
    g18 = count_smarts(mol, "[#6][CX3;!R](=O)[#6]")
    g19 = count_smarts(mol, "[#6][CX3;R](=O)[#6]")
    g20 = count_smarts(mol, "[CX3H1](=O)[#6]")
    g21 = count_smarts(mol, "[#6][CX3;!R](=O)[OX2][#6]")
    g22 = count_smarts(mol, "[#6;R][CX3;R](=O)[OX2;R][#6;R]")
    g23 = count_smarts(mol, "[CX3;!R](=O)[OX2H1]")
    g24 = count_smarts(mol, "[#6][OX2][CX3](=O)[OX2][#6]")
    g25 = count_smarts(mol, "[CX3](=O)O[CX3](=O)")
    g26 = count_smarts(mol, "[OX2H][OX2][#6]")
    g27 = count_smarts(mol, "[#6][OX2][OX2][#6]")
    return [g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g21,g22,g23,g24,g25,g26,g27]


HEADERS = [
    "1.1.dnAB1-branched,i","1.2.dnAB2-branched,i","1.3.dnAB3-branched,i","1.4.dnAB4-branched,i",
    "1.5.dnAB5-branched,i","1.6.dnAB6-branched,i","1.7.dnAB7-branched,i","1.8.dnAB8-branched,i",
    "1.9.dnAB9-branched,i","1.10.dnAB10-branched,i","1.11.dnAB11-branched,i","1.12.dnAB12-branched,i",
    "1.13.dnAB13-branched,i","1.14.dnAB14-branched,i","1.15.dnAB-unbranched,i","1.dnAB-sum",
    "2.dnCCDB,I, CH2=CH2, aromatic","3.dnCCDB,I, CH2=CH2, ring",
    "4.dnCCDB,I, CH2=CH2, non-aromatic, non-ring","5.dnCCTB,I, CC",
    "6.dnTC,i","7.dnQC,i","8.g-CH3,i","9.m(CH2),nonring","10.g-CH2-(nonring),i",
    "11.mCH2-(ring),i","12.g-CH2-(ring),i","13.CH, nonTC","14.C, nonQC",
    "15.g-OH,i","16.g-O-(nonring),i","17.g-O-(ring),i","18.g>C=O(nonring),i","19.g>C=O(ring),i","20.gO=CH-,i",
    "21.g-C=O-O-(nonring, ester),i","22.g-C=O-O-(ring, ester),i","23.g-C=O-OH(nonring, acid),i",
    "24.Carbonate ester","25.Carboxylic anhydride","26.Hydroperoxide","27.Peroxide",
]

def qspr_uob3_vector(mol, ab_smarts=None):
    v1 = ab14_counts(mol, ab_smarts_compiled=ab_smarts)
    v2 = groups_2_to_27(mol)
    return v1 + v2

def process_excel(in_path="all_data.xlsx",
                  out_dir="qspr_uob3_by_sheet",
                  ab_smarts_json=None):

    os.makedirs(out_dir, exist_ok=True)
    ab_map = load_ab_smarts(ab_smarts_json)

    xls = pd.ExcelFile(in_path)
    saved = []

    for sh in xls.sheet_names:
        print("=============={}==============".format(sh))
        df = pd.read_excel(in_path, sheet_name=sh)

        fuel_type_col = df.columns[0]
        label_col = df.columns[-2]



        vecs = []
        for smi in df["SMILES"].astype(str):
            mol = Chem.MolFromSmiles(smi.strip())
            vec = [None]*len(HEADERS) if mol is None else qspr_uob3_vector(mol, ab_map)
            vecs.append(vec)
        feats = pd.DataFrame(vecs, columns=HEADERS)


        out_df = pd.concat(
            [
                df[[fuel_type_col, label_col]].rename(
                    columns={fuel_type_col: "fuel_type", label_col: "label"}
                ),
                feats
            ],
            axis=1
        )


        csv_path = os.path.join(out_dir, f"{sh}.csv")
        pq_path  = os.path.join(out_dir, f"{sh}.parquet")
        out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")


        saved.append((sh, csv_path, pq_path))

    return saved


def quick_test():
    tests = {
        "Toluene": "Cc1ccccc1",
        "1-Methylnaphthalene": "Cc1cccc2ccccc12",
        "9-Methylanthracene": "Cc1ccc2cc3ccccc3cc2c1"
    }
    cols_show = ["1.1.dnAB1-branched,i","1.2.dnAB2-branched,i","1.3.dnAB3-branched,i",
                 "1.4.dnAB4-branched,i","1.5.dnAB5-branched,i","1.15.dnAB-unbranched,i",
                 "1.dnAB-sum","2.dnCCDB,I, CH2=CH2, aromatic",
                 "8.g-CH3,i","9.m(CH2),nonring","10.g-CH2-(nonring),i",
                 "11.mCH2-(ring),i","12.g-CH2-(ring),i"]
    rows=[]
    for name, smi in tests.items():
        mol = Chem.MolFromSmiles(smi)
        v = qspr_uob3_vector(mol, ab_smarts=None)
        row = {"name": name, "SMILES": smi}
        row.update({HEADERS[i]: v[i] for i in range(len(HEADERS))})
        rows.append({k: row.get(k, None) for k in (["name","SMILES"]+cols_show)})
    return pd.DataFrame(rows)

if __name__ == "__main__":

    process_excel("../data/revision/all_data.xlsx", out_dir="qspr_uob3_by_sheet", ab_smarts_json=None)

