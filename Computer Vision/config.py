import os
import torch
from enum import Enum

DATA_DIR = 'Images'
data_class2id_map = {"Classic": 0, "Modern": 1, "Soviet": 2 }
data_id2class_map = {0: "Classic", 1: "Modern", 2: "Soviet"}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReturnCodes(Enum):
    FAIL_WITH_ERROR = 0
    SUCCESS = 1
