# define training code
import csv
import gzip
import pickle
import pandas as pd
import SVM
from src.SVM import train_model


import numpy as np
import torch

# ~~~~~~~~~~~~~~~~~~~~~~~ General  Functions ~~~~~~~~~~~~~~~~~~~~~~~ #

def save_model(model, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(model, f)
        print("Saved model.")


def load_pca_model():
    with gzip.open('../models/PCA.pkl.gz', 'rb') as f:
        return pickle.load(f)


def pca_inverse_transform(pca, pca_100):
    return pca.inverse_transform(pca_100)


def load_and_concatenate_csvs(file_names, chunk_size=10000):
    df_list = []
    for file_name in file_names:
        input_file_path = '../datasets/' + file_name
        # Read the CSV file in chunks
        for chunk in pd.read_csv(input_file_path, compression='gzip', sep='|',chunksize=chunk_size):
            df_list.append(chunk)
    # Concatenate all chunks into a single DataFrame
    concatenated_df = pd.concat(df_list, ignore_index=True)
    return concatenated_df


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



# ~~~~~~~~~~~~~~~~~~~~~~~~ Kyle's Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #

def kyle_main():
    files = ['train_embeddings_0.csv.gz'  # ,
             # 'train_embeddings_1.csv.gz',
             # 'train_embeddings_2.csv.gz',
             # 'train_embeddings_3.csv.gz',
             # 'train_embeddings_4.csv.gz',
             # 'train_embeddings_5.csv.gz',
             # 'test_embeddings_0.csv.gz',
             # 'test_embeddings_1.csv.gz'
             ]
    train_data = load_and_concatenate_csvs(files)
    test_data = load_and_concatenate_csvs(['test_embeddings_0.csv.gz'])

    model = train_model(train_data, test_data)
    # save_model(model, "../models/SVM_PCA100.pkl.gz")

# ~~~~~~~~~~~~~~~~~~~~~~~~ Luke's Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
if __name__ == '__main__':


    # base_data = pd.read_csv('../datasets/', low_memory=False)

    # base_data = process_data(base_data) # preprocess

    NAME = 'K'

    match NAME:
        case 'J':
            pass
        case 'K':
            kyle_main()
        case 'L':
            pass
        case _:
            pass