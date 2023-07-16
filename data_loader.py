import os
import time

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, folder_path, label_path):
        self._folder_path = Path(folder_path)
        self._label_path = Path(label_path)
        self.data, self.label = self._get_dataset()

    def _get_dataset(self):
        '''return the dataset of the specific path'''
        # read dataset when {id} is in image channels path and label path meanwhile
        image_idx, label_idx, res = [], [], []
        image_dataset, label_dataset = [], []
        # available image idx
        for file_path in self._folder_path.iterdir():
            if file_path.is_file() and file_path.name[-10:] == '_mises.csv':
                image_idx += [int(file_path.stem[4:-6])]
        # available label idx
        for file_path in self._label_path.iterdir():
            if file_path.is_file() and file_path.name[-4:] == '.csv':
                label_idx += [int(file_path.stem[8:])]
        # available idx
        ava_idx = set(image_idx).intersection(set(label_idx))
        for idx in ava_idx:
            mises = pd.read_csv(self._folder_path.joinpath('void' + str(idx) + '_mises.csv'), header=None).values
            stress_x = pd.read_csv(self._folder_path.joinpath('void' + str(idx) + '_stress_x.csv'), header=None).values
            stress_y = pd.read_csv(self._folder_path.joinpath('void' + str(idx) + '_stress_y.csv'), header=None).values
            stress_z = pd.read_csv(self._folder_path.joinpath('void' + str(idx) + '_stress_z.csv'), header=None).values
            img = np.stack([mises, stress_x, stress_y, stress_z], axis=2)
            label = pd.read_csv(self._label_path.joinpath('voidlist' + str(idx) + '.csv'), header=None). \
                values.reshape(-1)
            image_dataset.append(img)
            label_dataset.append(label)
        return image_dataset, label_dataset

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).permute((2, 0, 1)).float(), \
            torch.from_numpy(self.label[idx]).float()


# 示例用法
folder_path = Path('./qfndata/inputs')
label_path = Path('./qfndata/outputs')
dataset = CustomDataset(folder_path, label_path)
