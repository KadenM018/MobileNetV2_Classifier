from typing import Dict, List, Any

import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

def add_args(parser):
    # input data info
    parser.add_argument('--data_path', type=str, default=os.path.join(os.getcwd(),"data"))
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=36)

    # hyperparameters
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=30)

    # other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_split', type=float, default=0.2)

    # save directories
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(os.getcwd(),"saves"))

    parser.add_argument('--bestonly', action='store_true')
    parser.add_argument('--name', type=str, default='exp')

    return parser.parse_args()


def map_label(in_label):
    mapping = {"0":0, "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
               "9":9, "A":10, "B":11, "C":12, "D":13, "E":14, "F":15, "G":16, "H":17,
               "I":18, "J":19, "K":20, "L":21, "M":22, "N":23, "O":24, "P":25, "Q":26,
               "R":27, "S":28, "T":29, "U":30, "V":31, "W":32, "X":33, "Y":34, "Z":35}
    assert in_label in mapping.keys(), "Incorrect labels for ASL Data: 0-9, A-Z."
    return mapping[in_label]


def create_datalist(dataset_path):
    paths = []
    labels = []
    filepath = Path(dataset_path)
    for root, dirs, files in sorted(os.walk(filepath)):
        for file in files:
            image_path = os.path.join(root, file)
            image_label = os.path.basename(root)
            paths.append(image_path)
            idx = map_label(image_label)
            labels.append(idx)
    return paths, labels


class ASLDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
