# define training code
import csv
import gzip
import os
import pickle
import pandas as pd
from torch import nn, optim
from tqdm import tqdm

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


def csv_file_to_nparray(file_name):
    rows = []
    file_path = '../datasets/' + file_name
    with gzip.open(file_path, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        next(reader, None)  # Skip the header row if it exists
        for row in reader:
            features = [float(value) for value in row[:100]]
            targets = [float(value) for value in row[100:104]]
            rows.append(features + targets)
    return np.array(rows)


def convert_to_cuda_tensor(np_array):
    return torch.tensor(np_array, device='cuda')

# ~~~~~~~~~~~~~~~~~~~~~~~~ Joe's  Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #

def joe_main():
    torch.set_default_dtype(torch.float32)

    model = ReviewNet()
    optimizer = optim.Adam(model.parameters())

    file_list = list([
        'train_embeddings_0.csv.gz' #,
        # 'train_embeddings_1.csv.gz',
        # 'train_embeddings_2.csv.gz',
        # 'train_embeddings_3.csv.gz',
        # 'train_embeddings_4.csv.gz',
        # 'train_embeddings_5.csv.gz'
    ])

    # Check if there's a checkpoint to resume from
    checkpoint_files = [f for f in os.listdir('../models') if f.startswith('checkpoint_')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        start_epoch, start_file_idx = load_checkpoint(model, optimizer, f'../models/{latest_checkpoint}')
        file_list = file_list[start_file_idx:]
    else:
        start_epoch = 0

    train_nn(model, optimizer, file_list, start_epoch)


class ReviewNet(nn.Module):
    def __init__(self):
        super(ReviewNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 42),
            nn.ReLU(),
            nn.Linear(42, 18),
            nn.ReLU(),
            nn.Linear(18, 4)
        )

    def forward(self, x):
        return self.layers(x)


def train_nn(model, optimizer, file_list, start_epoch=0, batch_size=32, num_epochs=10):
    criterion = nn.MSELoss()
    model.to('cuda')

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        for file_idx, file_name in enumerate(file_list):
            data = csv_file_to_nparray(file_name)
            np.random.shuffle(data)

            pbar = tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}, File {file_idx+1}/{len(file_list)}")
            for i in pbar:
                batch = data[i:i+batch_size]
                inputs = convert_to_cuda_tensor(batch[:, :100])
                targets = convert_to_cuda_tensor(batch[:, 100:])

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss / len(file_list):.4f}")

        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, f'../models/checkpoint_{epoch+1}.pth')

        # Save model after each epoch
        torch.save(model.state_dict(), f'../models/model_{epoch+1}.pth')


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['file_idx']

# ~~~~~~~~~~~~~~~~~~~~~~~~ Kyle's Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #

def kyle_main():
    files = [#'tweak_embeddings_0.csv.gz'
             'train_embeddings_0.csv.gz'
    #          'train_embeddings_1.csv.gz',
    #          'train_embeddings_2.csv.gz',
    #          'train_embeddings_3.csv.gz',
    #          'train_embeddings_4.csv.gz',
    #          'train_embeddings_5.csv.gz'
             ]
    train_data = load_and_concatenate_csvs(files)
    print("Files concatenated")
    test_data = load_and_concatenate_csvs(['test_embeddings_0.csv.gz'])

    model = train_model(train_data, test_data)
    save_model(model, "../models/SVM_PCA100.pkl.gz")

# ~~~~~~~~~~~~~~~~~~~~~~~~ Luke's Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
if __name__ == '__main__':


    # base_data = pd.read_csv('../datasets/', low_memory=False)

    # base_data = process_data(base_data) # preprocess

    NAME = 'K'

    match NAME:
        case 'J':
            joe_main()
        case 'K':
            kyle_main()
        case 'L':
            pass
        case _:
            pass