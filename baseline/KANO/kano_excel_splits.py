
import os
import json
from typing import Optional

import pandas as pd


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_smiles_col(df: pd.DataFrame) -> str:
    if 'SMILES' in df.columns:
        return 'SMILES'
    if 'smiles' in df.columns:
        return 'smiles'
    for c in df.columns:
        if str(c).lower() == 'smiles':
            return c
    raise ValueError("未找到 'SMILES' 列，请确认 all_data.xlsx 中存在该列。")


def _load_sheet(xlsx_path: str, sheet_name: Optional[str]):
    xl = pd.ExcelFile(xlsx_path)
    if sheet_name is None:
        sheet_name = xl.sheet_names[0]
    df = xl.parse(sheet_name)
    df = _normalize_cols(df)
    smiles_col = _find_smiles_col(df)
    if df.shape[1] < 3:
        raise ValueError(f"工作表 '{sheet_name}' 至少需要三列（首列类别、SMILES、以及倒数第二列 label）。")
    category_col = df.columns[0]
    label_col = df.columns[-2]
    return df, sheet_name, smiles_col, label_col, category_col


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def materialize_fold_to_csv(
    xlsx_path: str,
    sheet_name: Optional[str],
    split_json_path: str,
    fold_id: int,
    out_dir: str,
):

    df, sheet_name, smiles_col, label_col, _ = _load_sheet(xlsx_path, sheet_name)

    with open(split_json_path, 'r', encoding='utf-8') as f:
        sp = json.load(f)

    req = ['inner_train_idx', 'inner_val_idx', 'outer_test_idx']
    for k in req:
        if k not in sp:
            raise ValueError(f"split.json 缺少键：{k}（需要 {', '.join(req)}）")

    tr_ids = list(map(int, sp['inner_train_idx']))
    va_ids = list(map(int, sp['inner_val_idx']))
    te_ids = list(map(int, sp['outer_test_idx']))

    def _sub_to_csv(indices, path):
        sub = df.iloc[indices].copy()
        out = pd.DataFrame({
            'smiles': sub[smiles_col].astype(str).values,
            'target': pd.to_numeric(sub[label_col], errors='coerce').values
        })
        out.to_csv(path, index=False)

    fold_dir = os.path.join(out_dir, f'fold_{fold_id}')
    _ensure_dir(fold_dir)
    tr_p = os.path.join(fold_dir, "train.csv")
    va_p = os.path.join(fold_dir, "val.csv")
    te_p = os.path.join(fold_dir, "test.csv")

    _sub_to_csv(tr_ids, tr_p)
    _sub_to_csv(va_ids, va_p)
    _sub_to_csv(te_ids, te_p)

    return tr_p, va_p, te_p
