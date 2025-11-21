from enum import Enum
import os
import pandas as pd
import numpy as np
import collections.abc
from typing import Dict, Union, List, Tuple, Optional, cast
import math
import re

import torch
from torch.utils.data import Dataset
from torch_geometric.utils.smiles import from_smiles
from torch_geometric.data import Data
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from rdkit import Chem  # type: ignore



def parse_category(item: Union[str, float, None]) -> float:
    if isinstance(item, str):
        match = re.search(r'< (\d+(\.\d+)?)%', item)
        if match:
            return float(match.group(1))
    return np.nan


def stddev_serie_to_values(category_serie: pd.Series) -> np.ndarray:
    serie_perc = category_serie.apply(parse_category)
    serie_perc[serie_perc.isna()] = 100.0
    return (0.01 * serie_perc).to_numpy()


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr[np.newaxis, ...], dtype=torch.float32)


def adaptive_clip(vals: np.ndarray) -> np.ndarray:
    if np.isnan(vals).all():
        return vals
    masked = vals[~np.isnan(vals)]
    mean_v = float(np.mean(masked))
    perc_20 = float(np.percentile(masked, 20))
    thresh = max(0.2 * mean_v, perc_20)
    rectified = vals.copy()
    rectified[rectified < thresh] = thresh
    return rectified


ACCURATE_ERROR_PERC = {
    "MW": 1, "SSTD": 3, "TC": 3, "HFUS": 6, "PC": 3, "VC": 5, "ZC": 1, "ACEN": 2, "RG": 3,
    "TPT": 4, "TPP": 4, "SOLP": 4, "DM": 4, "LVOL": 2, "VDWV": 5, "HFOR": 6, "VDWA": 5,
    "GFOR": 6, "RI": 4, "ENT": 6, "HSUB": 3, "HSTD": 6, "PAR": 6, "GSTD": 6, "DC": 3,
    "ACCW": 6, "HLC": 6, "FLTL": 6, "FLTU": 6, "SOLW": 6, "LCP1": 3, "LCP2": 3, "LCP3": 3,
    "LCP4": 3, "LDN1": 3, "LDN2": 3, "LDN3": 3, "LDN4": 3, "LTC1": 3, "LTC2": 3, "LTC3": 3,
    "LTC4": 3, "LVS1": 3, "LVS2": 3, "LVS3": 3, "LVS4": 3, "ST1": 3, "ST2": 3, "ST3": 3,
    "ST4": 3, "VP1": 5, "VP2": 5, "VP3": 5, "VP4": 5, "VP5": 5, "VP6": 5, "CN": 5
}


def smiles_to_graph_opt(smiles: str) -> Data:
    return from_smiles(smiles)


class DatasetProcessor(Dataset, collections.abc.Sequence):

    property_full_names = {
        "MW": "Molecular Weight",

    }

    def benchmark_dataset_processor(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ChemID'] = 0 * len(df)
        df['Name'] = 'placeholder' * len(df)
        df['Formula'] = 'placeholder' * len(df)
        df['CASN'] = 'placeholder' * len(df)
        df['CNAM'] = 'placeholder' * len(df)
        df['INAM'] = 'placeholder' * len(df)
        df['Structure'] = 'placeholder' * len(df)
        df['Family'] = ['n-Alkanes'] * len(df)
        df['Sub_Family'] = 'placeholder' * len(df)
        df['STP State'] = ['L'] * len(df)
        df['MW'] = 0 * len(df)
        if 'smiles' in df.columns:
            df = df.rename(columns={'smiles': 'SMILES'})

        column_order = ['ChemID', 'Name', 'Formula', 'CASN', 'CNAM', 'INAM',
                        'SMILES', 'Structure', 'Family', 'Sub_Family', 'STP State', 'MW']
        new_columns = [c for c in df.columns if c not in column_order]

        for col in new_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[f"{col} Error [%]"] = "< 1%"
            df[f"{col} Data Type"] = "Experimental"
            df[f"{col} Notes"] = None
            df[f"{col} Ref#"] = None

        df = df[column_order + sum(([col] + [f"{col}{suf}" for suf in
                [" Error [%]", " Data Type", " Notes", " Ref#"]] for col in new_columns), [])]
        return df

    def __init__(self,
                 path: str,
                 tab: str,
                 task_type: str,
                 filters: Optional[str] = None,
                 cache_data: bool = False,
                 ablated_props: Optional[List[str]] = None) -> None:
        super().__init__()
        self.task_type = task_type


        if path.endswith('.csv'):
            df_orig = pd.read_csv(path)
        else:
            df_orig = pd.read_excel(path, tab)


        base_name = os.path.basename(path)
        self.non_saf_dataset = not base_name.upper().startswith('SAF')
        if self.non_saf_dataset:
            df_orig = self.benchmark_dataset_processor(df_orig)
            self.property_full_names = {col: col for col in df_orig.columns}
            print("Used dataset (non-SAF):", path)

        mol_weight_col = df_orig['MW']

        if ablated_props != 'None' and ablated_props is not None:
            if "LOW" in ablated_props:
                ablated_props = ["MW", "SSTD", "TC", "HFUS", "PC", "VC", "ZC", "ACEN", "RG",
                                 "TPT", "SOLP", "TPP", "DM", "LVOL", "VDWV", "HFOR", "VDWA",
                                 "GFOR", "RI", "...", "HSTD", "PAR", "GSTD", "ACCW", "HLC",
                                 "FLTL", "FLTU", "SOLW"]
            ap2 = ablated_props[:]
            for prop in ablated_props:
                ap2.append(prop + ' ')
            df_orig = df_orig.loc[:, ~df_orig.columns.str.contains(r'\b(?:' + '|'.join(ap2) + r')\b', case=False)]
            if "MW" not in df_orig.columns:
                df_orig.insert(loc=11, column='MW', value=mol_weight_col)


        feature_columns = list(df_orig.iloc[:, :12].columns)
        property_names = list(df_orig.iloc[:, 12::5].columns)


        clean_property_names = [p.strip() for p in property_names]


        series_smiles = df_orig['SMILES']
        series_formula = df_orig['Formula']
        series_family = df_orig['Family']
        series_state = df_orig['STP State']
        df_features = df_orig.iloc[:, :12]
        df_properties = df_orig[clean_property_names]
        df_error_categories = df_orig[[f"{p} Error [%]" for p in property_names]]
        df_property_stddevs = pd.DataFrame({
            p: stddev_serie_to_values(df_error_categories[f"{p} Error [%]"])
            for p in property_names
        })
        df_property_stddevs = df_property_stddevs.apply(adaptive_clip, axis=0)


        indices_to_remove: List[int] = []

        def _worker(args: Tuple[int, str]):
            idx, smiles = args
            if isinstance(smiles, float) and math.isnan(smiles):
                indices_to_remove.append(idx); return
            try:
                data_tmp = self.smiles_to_graph(smiles)
            except Exception:
                indices_to_remove.append(idx); return
            if (data_tmp is None) or (data_tmp.num_nodes == 0):
                indices_to_remove.append(idx)

        with ThreadPool(8) as pool:
            list(tqdm(pool.imap_unordered(_worker, series_smiles.items()), total=len(series_smiles)))

        if indices_to_remove:
            print(f'Faulty smiles treatment will remove {len(indices_to_remove)} indices!')
            series_smiles = series_smiles.drop(indices_to_remove)
            df_properties = df_properties.drop(indices_to_remove)
            df_error_categories = df_error_categories.drop(indices_to_remove)
            df_features = df_features.drop(indices_to_remove)
            series_formula = series_formula.drop(indices_to_remove)
            series_family = series_family.drop(indices_to_remove)

        assert len(series_smiles) > 0


        if filters is not None:
            filtered_indices = dataset_filter(filters, series_formula, series_smiles, series_family, series_state)
            print(f'Applying dataset filter {filters} removes {len(filtered_indices)} indices!')
            series_smiles = series_smiles.drop(filtered_indices)
            df_properties = df_properties.drop(filtered_indices)
            df_error_categories = df_error_categories.drop(filtered_indices)
            df_features = df_features.drop(filtered_indices)
            series_formula = series_formula.drop(filtered_indices)
            series_family = series_family.drop(filtered_indices)


        series_smiles = series_smiles.reset_index(drop=True)
        df_properties = df_properties.reset_index(drop=True)
        df_error_categories = df_error_categories.reset_index(drop=True)
        df_property_stddevs = df_property_stddevs.reset_index(drop=True)
        df_features = df_features.reset_index(drop=True)
        series_formula = series_formula.reset_index(drop=True)
        series_family = series_family.reset_index(drop=True)

        self.series_smiles = series_smiles
        self.series_formula = series_formula
        self.series_family = series_family
        self.df_features = df_features
        self.df_properties = df_properties
        self.df_error_categories = df_error_categories
        self.df_property_stddevs = df_property_stddevs
        self.features_column_name = list(df_features.columns)


        self.error_thresh_perc = (
            np.array([ACCURATE_ERROR_PERC[n] for n in clean_property_names], dtype=np.float32)
            if not self.non_saf_dataset else None
        )


        self._normalizer = Normalizer(self.df_properties.to_numpy())


        self.cache_path = f"data_cache/dataset_classic_{str(filters)}.pt"
        self.cached_data: Optional[Dict[int, Data]] = None
        if cache_data:
            self._create_or_load_cache()

    @property
    def normalizer(self) -> "Normalizer":
        return self._normalizer

    def _create_or_load_cache(self) -> None:
        if os.path.exists(self.cache_path):
            self.cached_data = torch.load(self.cache_path)
            return
        cached_data = {}
        for index in tqdm(range(len(self))):
            cached_data[index] = self._calc_item(index)
        os.makedirs(os.path.split(self.cache_path)[0], exist_ok=True)
        torch.save(cached_data, self.cache_path)
        self.cached_data = cached_data

    def __getitem__(self, index: Union[int, slice]) -> Data:
        if self.cached_data is not None:
            return self.cached_data[cast(int, index)]
        return self._calc_item(index)

    def _calc_item(self, index: Union[int, slice]) -> Data:
        assert isinstance(index, int)
        smiles = self.series_smiles[index]
        props = self.df_properties.iloc[index].to_numpy()
        stds = self.df_property_stddevs.iloc[index].to_numpy()
        row_err = self.df_error_categories.iloc[index]
        row_feat = self.df_features.iloc[index]

        data = self.smiles_to_graph(smiles)
        data.df_error_category_row = row_err
        data.df_features_row = row_feat
        data.properties = to_tensor(props)
        data.property_stddevs = to_tensor(stds)
        return data

    def __len__(self) -> int:
        return len(self.series_smiles)

    def smiles_to_graph(self, smiles: str) -> Data:
        return smiles_to_graph_opt(smiles)

    def keep_precise_only(self, data: Data) -> Data:
        if self.error_thresh_perc is None:
            return data
        result = data.__copy__()
        error_value = data.df_error_category_row.apply(parse_category).to_numpy()
        good_mask = error_value <= self.error_thresh_perc
        properties = data.properties.clone().squeeze(0)
        properties[~good_mask] = np.nan
        result.properties = properties.unsqueeze(0)
        return result


class Purpose(Enum):
    Train = "train"
    Val = "val"
    Test = "test"


class Split(Dataset, collections.abc.Sequence):
    def __init__(self, dataset: DatasetProcessor, indices: np.ndarray, task_type: str,
                 purpose: Purpose, oversample_factor: int = 1000):
        self.dataset = dataset
        self.indices = indices
        self.purpose = purpose
        self.oversample_factor = oversample_factor
        self.task_type = task_type

    def __getitem__(self, index: Union[int, slice]) -> Data:
        assert isinstance(index, int)
        idx = self.indices[index % len(self.indices)].item() if self.purpose == Purpose.Train else self.indices[index].item()
        data = self.dataset[idx]
        if self.purpose in (Purpose.Val, Purpose.Test):
            if (self.task_type == 'regression') and (self.dataset.error_thresh_perc is not None):
                data = self.dataset.keep_precise_only(data)
        return data

    def __len__(self) -> int:
        return len(self.indices) * self.oversample_factor if self.purpose == Purpose.Train else len(self.indices)

    def get_indices(self) -> np.ndarray:
        return self.indices

    def as_dataset(self) -> Dataset:
        return cast(Dataset, self)


class Normalizer:
    def __init__(self, props_matrix: np.ndarray):
        self.mean = np.nanmean(props_matrix, axis=0, keepdims=True)
        self.std = np.nanstd(props_matrix, axis=0, keepdims=True)
        self.std = np.where(self.std == 0, 1, self.std)

    def encode(self, props: np.ndarray) -> np.ndarray:
        return (props - self.mean) / self.std

    def decode(self, props: np.ndarray) -> np.ndarray:
        return props * self.std + self.mean


def dataset_filter(filters: Optional[str],
                   series_formula: pd.Series, series_smiles: pd.Series,
                   series_family: pd.Series, series_state: pd.Series) -> List[int]:
    if filters is None:
        filters = ""

    lowercase_filters = filters.lower()
    filters_list = [char for char in lowercase_filters]

    by_family = 'f' in filters_list
    by_atoms = 'a' in filters_list
    by_chain_length = 'c' in filters_list
    by_state = 's' in filters_list

    filtered_indices: List[int] = []
    permitted_atoms = ['C', 'H', 'O']

    if by_family:
        family_list = [
            'other polyfunctional c, h, o', 'silanes/siloxanes', 'sulfides/thiophenes',
            'polyols', 'c, h, f compounds', 'polyfunctional c, h, o, halide',
            'polyfunctional c, h, o, n', 'aliphatic ethers', 'other amines, imines',
            'aromatic amines', 'c, h, multihalogen compounds', 'aromatic esters',
            'polyfunctional amides/amines', 'other ethers/diethers', 'polyfunctional esters',
            'inorganic', 'other aldehydes', 'polyfunctional aldehydes', 'aliphatic acids',
            'aromatic acids', 'polyfunctional acids', 'aliphatic alcohols', 'aromatic alcohols',
            'polyfunctional alcohols', 'aliphatic esters', 'aromatic esters', 'polyfunctional esters',
            'aliphatic ketones', 'aromatic ketones', 'polyfunctional ketones', 'aliphatic alkanes',
            'cycloalkanes', 'aromatic alkanes', 'polyfunctional alkanes', 'aliphatic amides',
            'aromatic amides', 'polyfunctional amides', 'aliphatic amines', 'aromatic amines',
            'polyfunctional amines', 'aromatic ethers', 'polyfunctional ethers', 'aromatic halides',
            'polyfunctional halides'
        ]
        for idx, fam in series_family.items():
            if str(fam).lower() not in family_list:
                filtered_indices.append(idx)

    if by_atoms:
        for idx, smiles in series_smiles.items():
            mol = Chem.MolFromSmiles(smiles)  # type: ignore
            if not all(atom.GetSymbol() in permitted_atoms for atom in mol.GetAtoms()):
                filtered_indices.append(idx)

    if by_chain_length:
        min_length, max_length = 5, 20
        for idx, smiles in series_smiles.items():
            mol = Chem.MolFromSmiles(smiles)  # type: ignore
            chain_length = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
            if (chain_length < min_length) or (chain_length > max_length):
                filtered_indices.append(idx)

    if by_state:
        for idx, state in series_state.items():
            if state != 'L':
                filtered_indices.append(idx)

    return list(set(filtered_indices))
