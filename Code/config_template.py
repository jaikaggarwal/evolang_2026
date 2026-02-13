from pydantic import BaseModel
from typing import List
import yaml


class IBParameters(BaseModel):
    weights: List[float]
    mu: float


class ILConfig(BaseModel):

    domain: str
    data_loader_name: str
    data_loader_params: dict
    use_presaved_data_loader: bool

    need_param: str
    model_params: IBParameters
    relation_params: dict
    include_only_convex_clusterings: bool

    max_beta: float



def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
        return data