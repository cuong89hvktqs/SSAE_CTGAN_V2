import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import MAECTGANSynthesizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sdv.errors import  InvalidDataTypeError
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from process_data import load_data_unsw,load_data_cic_ids,load_data_nsl_kdd
from sklearn.model_selection import train_test_split
import sdv
from process_data import save_per_attack_cic_ids
print(sdv.__file__)

import ctgan
print(ctgan.__file__)



if __name__ == "__main__":
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
    print("Torch version:", torch.__version__)


    
