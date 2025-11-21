
import numpy as np
if not hasattr(np, 'float'):  np.float = float
if not hasattr(np, 'int'):    np.int = int
if not hasattr(np, 'bool'):   np.bool = bool
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'str'):    np.str = str
import sys
import warnings
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from mordred import Calculator, descriptors
from mordred.error import Missing
from mordred.error import Missing, Error



def to_float_or_zero(v):
    if isinstance(v, (Missing, Error)):
        return 0.0
    try:
        f = float(v)
    except Exception:
        return 0.0
    import numpy as _np
    if not _np.isfinite(f):
        return 0.0
    return f


def make_mol_from_smiles(smiles, add_h=True, embed_3d=True):
    if smiles is None:
        return None
    s = str(smiles).strip()
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


def descriptor_objects_and_names(calc):

    m = make_mol_from_smiles("C", add_h=True, embed_3d=True)
    res = calc(m)
    desc_objs = []
    for k in res.keys():
        desc_objs.append(k)

    desc_names = []
    i = 0
    while i < len(desc_objs):
        name = str(desc_objs[i])
        desc_names.append(name)
        i += 1

    seen = {}
    i = 0
    while i < len(desc_names):
        n = desc_names[i]
        if n in seen:
            seen[n] += 1
            new_name = n + "__" + str(seen[n])
            desc_names[i] = new_name
        else:
            seen[n] = 1
        i += 1

    return desc_objs, desc_names




def compute_descriptors_for_list(smiles_list, calc, desc_objs):

    n = len(smiles_list)
    d = len(desc_objs)
    out = np.zeros((n, d), dtype=np.float32)

    i = 0
    while i < n:
        smi = smiles_list[i]
        mol = make_mol_from_smiles(smi, add_h=True, embed_3d=True)
        if mol is None:
            i += 1
            continue

        try:
            res = calc(mol)
        except Exception:
            i += 1
            continue

        j = 0
        while j < d:
            dobj = desc_objs[j]
            try:
                v = res[dobj]
            except Exception:
                v = Missing()
            out[i, j] = to_float_or_zero(v)
            j += 1

        i += 1

    return out


def process_workbook(input_xlsx, output_xlsx, label_index):
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    calc = Calculator(descriptors, ignore_3D=False)
    desc_objs, desc_names = descriptor_objects_and_names(calc)

    xls = pd.ExcelFile(input_xlsx)
    writer = pd.ExcelWriter(output_xlsx, engine="openpyxl", mode="w")

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        print("=================={}====================".format(sheet))
        if "SMILES" in df.columns:
            smiles_col = "SMILES"
        else:
            smiles_col = None
            for c in df.columns:
                if str(c).strip().lower() == "smiles":
                    smiles_col = c
                    break
        if smiles_col is None:
            print("[WARN] Sheet '" + str(sheet) + "' skipped (no 'SMILES' column).")
            continue

        if df.shape[1] < 2:
            print("[WARN] Sheet '" + str(sheet) + "' skipped (needs at least 2 columns).")
            continue

        label_series = df.iloc[:, label_index].copy()
        label_series.name = "label"

        smiles_list = df[smiles_col].astype(str).tolist()

        X = compute_descriptors_for_list(smiles_list, calc, desc_objs)

        feats_df = pd.DataFrame(X, columns=desc_names)
        feats_df = feats_df.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)

        out_df = pd.concat([label_series.reset_index(drop=True), feats_df], axis=1)
        out_df.to_excel(writer, sheet_name=sheet, index=False)

    writer.close()
    print("Done. Saved descriptor workbook to: " + str(output_xlsx))


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage:")
        print("  python build_descriptors_simple.py <input_all_data.xlsx> [output.xlsx]")
        print("Example:")
        print("  python build_descriptors_simple.py all_data.xlsx descriptors_all.xlsx")
        return 2

    input_xlsx = sys.argv[1]
    if len(sys.argv) == 3:
        output_xlsx = sys.argv[2]
    else:
        output_xlsx = "descriptors_all.xlsx"

    process_workbook(input_xlsx, output_xlsx)
    return 0



if __name__ == "__main__":

    process_workbook("../data/revision/all_data.xlsx", "all_descriptors.xlsx", -1)
