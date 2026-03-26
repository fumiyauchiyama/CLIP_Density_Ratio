import os
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import csv
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def dump_config(config, path):
    yaml_dump = OmegaConf.to_yaml(OmegaConf.structured(config))
    with open(path, "w") as f:
        f.write(yaml_dump)


def save_csv(data: Dict[str, List[Any]], filename: str, encoding: str = 'utf-8') -> None:

    lengths = [len(v) for v in data.values()]
    if not lengths or any(l != lengths[0] for l in lengths):
        raise ValueError("All lists in the dictionary must have the same length.")
    
    with open(filename, 'w', newline='', encoding=encoding) as f:
        writer = csv.writer(f)
        
        header = list(data.keys())
        writer.writerow(header)
        
        for row in zip(*data.values()):
            writer.writerow(row)

