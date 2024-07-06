import torch
import pandas as pd
import os


def drop_null_obj(y: torch.tensor):
    null = (y == 0).all(dim=0)
    mask = (y != 0).any(dim=0)
    # Use the mask to select columns that are not all zeros
    filtered_tensor = y[:, mask]
    return filtered_tensor, null


def pv_loads_to_input(feeder):
    loads_file = os.path.join(os.path.dirname(__file__), feeder, 'loads.csv')
    capacity_file = os.path.join(os.path.dirname(__file__), feeder, 'capacity.csv')

    pv_loads = pd.read_csv(loads_file)
    pv_loads["name"] = pv_loads["name"].map(lambda x: x.strip("pv_"))
    pv_caps = pd.read_csv(capacity_file)

    pv_scens = pd.pivot_table(pv_loads, index=['name', "busName"], columns='scenario', aggfunc=len, fill_value=0)
    if len(pv_scens) != len(pv_caps):
        raise Exception("Some adopters never adopt. Design matrix is missing rows!")
    x = torch.tensor(pv_scens.T.values, dtype=torch.float64)
    if (x > 1).any():
        raise Exception("Ill-defined design matrix.")
    return pv_scens.T, x