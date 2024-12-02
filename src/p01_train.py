# define training code
import csv
import gzip
import os
import pickle
import pandas as pd
from torch import nn, optim
from tqdm import tqdm

import SVM
from src.NeuralNets import ReviewNet, ReviewNetLarge, ReviewNetLargeNorm
from src.SVM import train_model, use_model

import numpy as np
import torch

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

torch.set_default_dtype(torch.float32)

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


def load_and_concatenate_csvs(file_names, chunk_size=10_000):
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
    return torch.tensor(np_array, device='cuda', dtype=torch.float32)

# ~~~~~~~~~~~~~~~~~~~~~~~~ Joe's  Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #

def joe_main():
    model = ReviewNetLarge(dropout_rate=0.2)
    optimizer = optim.Adam(model.parameters())

    file_list = [
        'train_embeddings_0.csv.gz',
        'train_embeddings_1.csv.gz',
        'train_embeddings_2.csv.gz',
        'train_embeddings_3.csv.gz',
        'train_embeddings_4.csv.gz',
        'train_embeddings_5.csv.gz'
    ]

    # Check if there's a checkpoint to resume from
    checkpoint_files = [f for f in os.listdir('../models') if f.startswith('checkpoint_')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        checkpoint = torch.load(f'../models/{latest_checkpoint}', map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to('cuda')  # Move model to GPU
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()  # Move optimizer state to GPU
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # torch.save(model.state_dict(), f'../models/best_model_nn2.pth')

    train_nn(model, optimizer, file_list, start_epoch, patience=5)


# Modified/made more complex for Model 2+
def train_nn(model, optimizer, file_list, start_epoch=0, batch_size=64, num_epochs=300, patience=5):
    criterion = ordinal_loss
    model.to('cuda')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_loss = float('inf')
    patience_counter = 0
    best_model = None

    data = None
    for filename in file_list:
        file_data = csv_file_to_nparray(filename)
        if data is None:
            data = file_data
        else:
            data = np.concatenate((np.array(data), np.array(file_data)), axis=0)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        n_iter = 0

        np.random.shuffle(data)

        pbar = tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")
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
            n_iter += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / n_iter
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Save checkpoint after processing all files in the epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }
        torch.save(checkpoint, f'../models/checkpoint_epoch_{epoch+1}.pth')

    # Save the best model
    if best_model is not None:
        torch.save(best_model, f'../models/best_model_nn2.pth')


def ordinal_loss(predictions, targets):
    stars_pred = predictions[:, 0]
    stars_true = targets[:, 0]
    other_pred = predictions[:, 1:]
    other_true = targets[:, 1:]

    # MSE for other outputs
    mse_loss = nn.MSELoss()(other_pred, other_true)

    # Ordinal regression loss for stars
    levels = torch.arange(6).float().to(predictions.device)
    diff = (stars_pred.unsqueeze(1) - levels.unsqueeze(0)).abs()
    ord_loss = torch.mean((diff - (stars_true.unsqueeze(1) - levels.unsqueeze(0)).abs()).pow(2))

    return mse_loss + ord_loss


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['file_idx']

# ~~~~~~~~~~~~~~~~~~~~~~~~ Kyle's Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #

def kyle_main():
    files = [#'tweak_embeddings_0.csv.gz'
             'train_embeddings_0.csv.gz',
             'train_embeddings_1.csv.gz',
             'train_embeddings_2.csv.gz',
             'train_embeddings_3.csv.gz',
             'train_embeddings_4.csv.gz',
             'train_embeddings_5.csv.gz'
             ]
    train_data = load_and_concatenate_csvs(files)
    test_data = load_and_concatenate_csvs(['test_embeddings_0.csv.gz'])
    print("Files concatenated")

    model = train_model(train_data)
    use_model(model, test_data)
    save_model(model, "../models/SVM_PCA100.pkl.gz")

# ~~~~~~~~~~~~~~~~~~~~~~~~ Luke's Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #


def lukasz_main():
    train_files = ['train_embeddings_0.csv.gz']
                   # 'train_embeddings_2.csv.gz',
                   # 'train_embeddings_1.csv.gz',
                   # 'train_embeddings_3.csv.gz',
                   # 'train_embeddings_4.csv.gz',
                   # 'train_embeddings_5.csv.gz']
    test_files = ['test_embeddings_0.csv.gz']  # 'test_embeddings_1.csv.gz']

    train_data = load_and_concatenate_csvs(train_files)
    test_data = load_and_concatenate_csvs(['test_embeddings_0.csv.gz'])
    model = lukasz_train(train_data, test_data)
    save_model(model, "../models/NB_PCA100.pkl.gz")


def lukasz_train(train_data: pd.DataFrame, test_data: pd.DataFrame):
    target_names = ['stars', 'useful', 'funny', 'cool']
    train_data_y = train_data[target_names].values
    train_data_x = train_data.drop(['stars', 'useful', 'funny', 'cool'], axis=1).values

    test_data_y = test_data[['stars', 'useful', 'funny', 'cool']].values
    test_data_x = test_data.drop(['stars', 'useful', 'funny', 'cool'], axis=1).values

    # train model
    model = MultiOutputClassifier(CustomGaussianNB())
    model.fit(train_data_x, train_data_y)

    # predict
    y_pred = model.predict(test_data_x)

    # Evaluate
    print("Classification Report:")
    for i, target in enumerate(['stars', 'useful', 'funny', 'cool']):
        print(f"\nTarget: {target}")
        print(classification_report(test_data_y[:, i], y_pred[:, i]))

    return model


class CustomGaussianNB(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.class_stats = {}

    def fit(self, x, y):
        """
        Train the Naive Bayes classifier.
        X: Feature matrix (2D array)
        y: Target vector (1D array)
        """
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            X_cls = x[y == cls]
            self.class_stats[cls] = {
                "mean": np.mean(X_cls, axis=0),
                "var": np.var(X_cls, axis=0) + 1e-6,  # Add small value to avoid division by zero
                "prior": len(X_cls) / len(x),
            }
        return self

    def predict(self, x):
        """
        Predict the class for each sample in X.
        X: Feature matrix (2D array)
        """
        log_posteriors = []
        debug_sample_loops = 0
        for cls, stats in self.class_stats.items():
            mean, var, prior = stats["mean"], stats["var"], stats["prior"]
            print(debug_sample_loops)

            # Vectorized computation for likelihood
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))  # Normalization
            log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / var, axis=1)  # Exponent

            # Add prior
            log_posterior = np.log(prior) + log_likelihood
            log_posteriors.append(log_posterior)

            debug_sample_loops += 1

        # Combine all class probabilities and choose the best
        log_posteriors = np.array(log_posteriors).T  # Transpose for (samples, classes)
        return np.argmax(log_posteriors, axis=1)  # Class with max posterior


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
            lukasz_main()
        case _:
            pass