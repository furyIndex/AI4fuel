
import os, glob, argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from mordred.error import Missing, Error


if not hasattr(np, 'float'):  np.float  = float  # type: ignore
if not hasattr(np, 'int'):    np.int    = int    # type: ignore
if not hasattr(np, 'bool'):   np.bool   = bool   # type: ignore
if not hasattr(np, 'object'): np.object = object # type: ignore
if not hasattr(np, 'str'):    np.str    = str    # type: ignore


def to_float_or_zero(v):
    if isinstance(v, (Missing, Error)):
        return 0.0
    try:
        f = float(v)
    except Exception:
        return 0.0
    if not np.isfinite(f):
        return 0.0
    return f


def make_mol_from_smiles(smi: str, add_h=True, embed_3d=True):
    if smi is None:
        return None
    s = str(smi).strip()
    if s == "":
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    if add_h:
        mol = Chem.AddHs(mol)
    if embed_3d:
        try:
            code = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if code == 0:
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                except Exception:
                    pass
        except Exception:
            pass
    return mol


def build_descriptor_template(include_3d: bool):
    calc = Calculator(descriptors, ignore_3D=(not include_3d))
    m = make_mol_from_smiles("C", add_h=True, embed_3d=include_3d)  # 甲烷
    res = calc(m)
    desc_objs = list(res.keys())
    names = []
    seen = {}
    for obj in desc_objs:
        n = str(obj)
        if n in seen:
            seen[n] += 1
            n = f"{n}__{seen[n]}"
        else:
            seen[n] = 1
        names.append(n)
    return calc, desc_objs, names


def compute_descriptors(smiles_list, calc, desc_objs, include_3d: bool):
    n, d = len(smiles_list), len(desc_objs)
    X = np.zeros((n, d), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = make_mol_from_smiles(smi, add_h=True, embed_3d=include_3d)
        if mol is None:
            continue
        try:
            res = calc(mol)
        except Exception:
            res = {}
        for j, dobj in enumerate(desc_objs):
            try:
                v = res[dobj]
            except Exception:
                v = Missing()
            X[i, j] = to_float_or_zero(v)
        if (i + 1) % 1000 == 0:
            print(f"  processed {i+1}/{n}")
    return X


def process_csv(in_csv: str, out_csv: str, include_3d: bool):
    print(f"\n=== Processing {os.path.basename(in_csv)} ===")
    df = pd.read_csv(in_csv)

    if df.shape[1] < 2:
        raise ValueError(f"{in_csv}: At least 2 columns are required (the first column is SMILES, and the second column onwards are labels).")


    smiles_col = df.columns[0]
    label_cols = list(df.columns[1:])


    print(f"  SMILES column: {smiles_col}")
    print(f"  Label columns: {label_cols}")

    labels_df = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    calc, desc_objs, desc_names = build_descriptor_template(include_3d=include_3d)
    smiles_list = df[smiles_col].astype(str).tolist()
    X = compute_descriptors(smiles_list, calc, desc_objs, include_3d=include_3d)

    feats_df = pd.DataFrame(X, columns=desc_names).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


    out_df = pd.concat([labels_df.reset_index(drop=True), feats_df], axis=1)
    out_df.rename(columns={c: f"label__{c}" for c in label_cols}, inplace=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv} (rows={len(out_df)}, feats={feats_df.shape[1]}, labels={len(label_cols)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory with MoleculeNet CSVs (first col = SMILES)")
    ap.add_argument("--out_dir", required=True, help="Directory to save *_desc.csv")
    ap.add_argument("--only2d", action="store_true", help="Only compute 2D descriptors (faster)")
    args = ap.parse_args()

    include_3d = not args.only2d
    csvs = sorted(glob.glob(os.path.join(args.in_dir, "*.csv")))
    if not csvs:
        print(f"No CSV found under {args.in_dir}")
        return

    for f in csvs:
        out_csv = os.path.join(args.out_dir, os.path.basename(f).replace(".csv", "_desc.csv"))
        try:
            process_csv(f, out_csv, include_3d=include_3d)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(f)}: {e}")

if __name__ == "__main__":
    main()
