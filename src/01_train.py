# define training code
import csv
import gzip
import pickle

import numpy as np
import torch

# ~~~~~~~~~~~~~~~~~~~~~~~ General  Functions ~~~~~~~~~~~~~~~~~~~~~~~ #

def load_pca_model():
    with gzip.open('../models/PCA.pkl.gz', 'rb') as f:
        return pickle.load(f)


def pca_inverse_transform(pca, pca_100):
    return pca.inverse_transform(pca_100)


def csv_files_to_nparray(file_list):
    rows = []
    for file_name in file_list:
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Convert row to float values and append to rows list
                rows.append([float(value) for value in row])

    # Convert the list of rows to a 2D NumPy array
    return np.array(rows)


def convert_to_cuda_tensor(np_array):
    return torch.tensor(np_array, device='cuda')


# ~~~~~~~~~~~~~~~~~~~~~~~~ Joe's  Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
if __name__ == '__main__':

    # base_data = pd.read_csv('../datasets/', low_memory=False)

    # base_data = process_data(base_data) # preprocess

    NAME = 'K'

    match NAME:
        case 'J':
            pass
        case 'K':
            pass
        case 'L':
            pass
        case _:
            pass