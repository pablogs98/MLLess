# Logistic Regression on Criteo
import pickle
from csv import reader, writer

from tqdm import tqdm
import mmh3

from MLLess.utils.preprocessing import Preprocessing

HASH_SIZE = 10 ** 5


# result is a list of tuples (label, list of tuples (index, value))
# Load a CSV file
def load_csv_sparse_tab(filename):
    with open(filename, 'rb') as file:
        num_lines = Preprocessing._line_count(file)

    progress = tqdm(total=num_lines, unit="lines",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    with open('criteo/sparse_criteo.pickle', 'wb') as intermediate_file:
        with open(filename, 'r') as file:
            csv_reader = reader(file, dialect="excel-tab")
            for row in csv_reader:
                tuples_list = []

                # add numerical values to list
                for i in range(1, 14):
                    if row[i] == '':
                        row[i] = "0"
                    tuples_list.append((i - 1, float(row[i])))

                global HASH_SIZE
                dic = {}
                for i in range(14, len(row)):
                    if row[i] == '':
                        continue
                    hashed = mmh3.hash(row[i], 42)
                    hashed = hashed % HASH_SIZE + 14

                    if hashed in dic:
                        dic[hashed] += 1
                    else:
                        dic[hashed] = 1

                for i in dic.keys():
                    tuples_list.append((i, float(dic[i])))

                row_entry = (float(row[0]), tuples_list)
                pickle.dump(row_entry, intermediate_file)
                progress.update()


# Find the min and max values for each column
def dataset_minmax():
    col_values = dict()
    col_counts = dict()
    col_min_max = dict()

    with open('criteo/sparse_criteo.pickle', 'rb') as file:
        while True:
            try:
                row = pickle.load(file)
                row_values = row[1]
                for i in range(len(row_values)):
                    index = row_values[i][0]
                    value = row_values[i][1]

                    if index in col_values:
                        col_values[index] += value
                        col_counts[index] += 1
                        col_min_max[index] = ( \
                            min(col_min_max[index][0], value), \
                            max(col_min_max[index][1], value))
                    else:
                        col_values[index] = value
                        col_counts[index] = 1
                        col_min_max[index] = (value, value)
            except EOFError:
                break
    return col_min_max


# Rescale dataset columns to the range 0-1
def normalize_dataset(minmax):
    with open('criteo/sparse_criteo.pickle', 'rb') as picklefile:
        with open('criteo/sparse_criteo_scale.csv', 'w', newline='') as csvfile2:
            csv_writer = writer(csvfile2)
            while True:
                try:
                    row = pickle.load(picklefile)
                    row_values = row[1]
                    write_row = [row[0]]
                    for i in range(len(row_values)):
                        index = row_values[i][0]
                        value = row_values[i][1]

                        if minmax[index][0] == minmax[index][1]:
                            continue
                        new_value = (value - minmax[index][0]) \
                                    / (minmax[index][1] - minmax[index][0])

                        if new_value != 0:
                            write_row.append(row_values[i][0])
                            write_row.append(new_value)
                    csv_writer.writerow(write_row)
                except EOFError:
                    break
