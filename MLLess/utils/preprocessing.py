import csv
import os
import pickle
import zlib
from enum import Enum
import random

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas

import numpy as np


class DatasetTypes(Enum):
    CRITEO = 0
    MOVIELENS = 1


class WrongNumberOfFeatures(Exception):
    pass


class Preprocessing:
    def __init__(self, dataset_path, dataset_type, separator, n_features=-1, sparse=False):
        if dataset_type == DatasetTypes.CRITEO and n_features <= 0:
            raise WrongNumberOfFeatures("Datasets with binary classes should have at least 1 feature")

        self.dataset_path = dataset_path
        self.separator = separator
        self.n_features = n_features
        self.has_class = dataset_type == DatasetTypes.CRITEO
        self.sparse = sparse

    # minibatch_size can't be greater than lines / n_workers

    def partition(self, minibatch_size, key, compress=True, scale=True):
        if not self.has_class:
            df = pandas.read_csv(self.dataset_path, sep=self.separator, header=None)
            dataset_mean = np.float64(df.iloc[:, 2].mean())
            df = df.sample(frac=1)
            df.to_csv("shuffle.csv", header=None, index=False, columns=[0, 1, 2])
            self.separator = ','
        else:
            if self.sparse:
                if not os.path.isfile('shuffle.csv'):
                    with open(self.dataset_path, 'r') as source:
                        data = [(random.random(), line) for line in source]
                    data.sort()
                    with open('shuffle.csv', 'w') as target:
                        for _, line in data:
                            target.write(line)
                else:
                    print("Found shuffle.csv, skipping shuffling.")
            else:
                df = pandas.read_csv(self.dataset_path, sep=self.separator, header=None,
                                     usecols=list(range(0, self.n_features + 1)))
                df = df.sample(frac=1)
                if scale:
                    print("Normalising...")
                    df.iloc[:, 1:self.n_features + 1] = StandardScaler().fit_transform(
                        df.iloc[:, 1:self.n_features + 1])
                df.to_csv("shuffle.csv", header=None, index=False)
            self.separator = ','

        with open(self.dataset_path, 'rb') as file:
            num_lines = Preprocessing._line_count(file)
        num_minibatches = int(num_lines / minibatch_size)
        mod_minibatches = num_lines % minibatch_size

        progress = tqdm(total=int(num_lines / minibatch_size), unit="parts",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

        with open("shuffle.csv") as file:
            print("Open dataset file. Processing {} lines. Creating {} minibatches".format(num_lines, num_minibatches))
            samples = []
            row_scale = {}
            col_scale = {}
            cols = 0
            rows = 0
            current_mb_line = 0
            part = 0

            with progress:
                for i, line in enumerate(file):
                    samples.append(self._parse(line))
                    current_mb_line += 1
                    mb_size = minibatch_size + 1 if part < mod_minibatches else minibatch_size
                    if current_mb_line == mb_size:
                        current_mb_line = 0
                        with open("/Users/pablo/PycharmProjects/MLLess/datasets/"
                                  "dataset_parts/{}-part{}.pickle".format(key, part), "wb") as f:
                            if self.has_class:
                                if self.sparse:
                                    labels = []
                                    smp = []
                                    for sample in samples:
                                        l = sample[0]
                                        labels.append(l)
                                        s = sample[1]
                                        smp.append(s)
                                    data = (np.array(labels), smp)
                                else:
                                    data = np.zeros((len(samples), self.n_features + 1), dtype=np.float64)
                                    for c, sample in enumerate(samples):
                                        data[c] = sample
                            else:
                                data = np.zeros((len(samples), 3), dtype=np.float64)
                                for n, (row, col, rating) in enumerate(samples):

                                    if scale:
                                        if col_scale.get(col, None) is None:
                                            col_scale[col] = cols
                                            cols += 1
                                        if row_scale.get(row, None) is None:
                                            row_scale[row] = rows
                                            rows += 1
                                    else:
                                        row_scale[row] = row
                                        col_scale[col] = col
                                    data[n, 0] = row_scale[row]
                                    data[n, 1] = col_scale[col]
                                    data[n, 2] = rating - dataset_mean
                            p = pickle.dumps(data)
                            if compress:
                                p = zlib.compress(p, 3)
                            f.write(p)
                        progress.update()
                        part += 1
                        samples = []
        try:
            pass
            # os.remove('shuffle.csv')
        except Exception as e:
            print(e)

        if not self.has_class:
            print("Dataset mean={}".format(dataset_mean))
        return part

    def generate_netflix_dataset(self):
        output = csv.writer(open('../../datasets/netflix/dataset.csv', 'w'))
        for i in range(4):
            with open(f'../../datasets/netflix/combined_data_{i + 1}.txt', 'r') as f:
                for line in f:
                    line = line.strip().rstrip('\n')
                    if ':' in line:
                        item = line.replace(':', '')
                    else:
                        user, rating, _ = line.split(',')
                        row = [int(user) - 1, int(item) - 1, rating]
                        output.writerow(row)

    @staticmethod
    def _line_count(file):
        f_gen = Preprocessing._make_gen(file.read)
        return sum(buf.count(b'\n') for buf in f_gen)

    @staticmethod
    def _make_gen(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)

    def _parse(self, line):
        features = []
        label = 0.0

        if line != '':
            line.rstrip()
            line = line.replace('"', '')
            line = line.split(self.separator)

            if self.has_class:
                if self.sparse:
                    idx = []
                    vals = []
                    for j in range(1, len(line)):
                        if j % 2 == 1:
                            idx.append(float(line[j]))
                        else:
                            vals.append(float(line[j]))
                    features = np.zeros((len(idx), 2))
                    for i in range(len(idx)):
                        if vals[i] != 0:
                            features[i, 0] = idx[i]
                            features[i, 1] = vals[i]
                else:
                    for j in range(1, self.n_features + 1):
                        f = line[j]
                        if f == '':
                            f = float(0)
                        else:
                            try:
                                f = float(f)
                            except ValueError:
                                f = float(0)
                        features.append(f)
                label = float(line[0])
            else:
                features.append(float(line[len(line) - 1]))

        if self.has_class:
            if self.sparse:
                return label, features
            else:
                features.append(label)
                return np.array(features, dtype=np.float64)
        else:
            return int(line[0]), int(line[1]), np.float64(line[2])
