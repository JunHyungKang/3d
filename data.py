import numpy as np
import random
import math
import os
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import time


def get_vector(points, size=16):
    x_y_z = [size, size, size]
    # 3D Points -> [16,16,16]
    xyzmin = np.min(points, axis=0) - 0.001
    xyzmax = np.max(points, axis=0) + 0.001

    diff = max(xyzmax - xyzmin) - (xyzmax - xyzmin)
    xyzmin = xyzmin - diff / 2
    xyzmax = xyzmax + diff / 2
    segments = []
    shape = []

    for i in range(3):
        # note the +1 in num
        if type(x_y_z[i]) is not int:
            raise TypeError("x_y_z[{}] must be int".format(i))
        s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
        segments.append(s)
        shape.append(step)

    n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]
    n_x = x_y_z[0]
    n_y = x_y_z[1]
    n_z = x_y_z[2]

    structure = np.zeros((len(points), 4), dtype=int)
    structure[:, 0] = np.searchsorted(segments[0], points[:, 0]) - 1
    structure[:, 1] = np.searchsorted(segments[1], points[:, 1]) - 1
    structure[:, 2] = np.searchsorted(segments[2], points[:, 2]) - 1

    # i = ((y * n_x) + x) + (z * (n_x * n_y))
    structure[:, 3] = ((structure[:, 1] * n_x) + structure[:, 0]) + (structure[:, 2] * (n_x * n_y))

    vector = np.zeros(n_voxels)
    count = np.bincount(structure[:, 3])
    vector[:len(count)] = count
    vector = vector.reshape((n_z, n_y, n_x))
    return vector


class CustomDataset(Dataset):
    def __init__(self, id_list, label_list, augment, task, args):
        self.id_list = id_list
        self.label_list = label_list
        self.augment = augment
        self.args = args
        self.task = task
        if task == 'trainval':
            self.pre_load = {x: np.load(os.path.join(self.args.trainval_data, f'{x}.npy')) for x in
                             self.id_list}
        elif task == 'test':
            self.pre_load = {x: np.load(os.path.join(self.args.test_data, f'{x}.npy')) for x in
                             self.id_list}

    def __getitem__(self, index):
        image_id = self.id_list[index]
        points = self.pre_load[image_id]
        # if self.task == 'trainval':
        #     points = os.path.join(self.args.trainval_data, f'{image_id}.npy')
        # elif task == 'test':
        #     points = os.path.join(self.args.test_data, f'{image_id}.npy')

        if self.augment:
            points = rand_rotate(points)
        image = get_vector(points, self.args.input_size)
        # print(np.max(image), np.min(image))

        if self.label_list is not None:
            label = self.label_list[index]
            return torch.Tensor(image).unsqueeze(0), label
        else:
            return torch.Tensor(image).unsqueeze(0)

    def __len__(self):
        return len(self.id_list)


def rand_rotate(dots):
    a, b, c = math.radians(random.randint(0, 360)), math.radians(random.randint(0, 360)), math.radians(random.randint(0, 360))
    mx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    my = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    mz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    m = np.dot(np.dot(mx, my), mz)
    dots = np.dot(dots, m.T)
    return dots
