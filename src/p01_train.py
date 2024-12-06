# define training code
import csv
import gzip
import os
import pickle
from pydoc import classname

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from torch import nn, optim
from tqdm import tqdm

import SVM
from src.NeuralNets import ReviewNet, ReviewNetLarge, ReviewNetLargeNorm, ReviewNetSmall, ReviewNetBERT
from src.SVM import train_model, use_model

import numpy as np
import torch

from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler


torch.set_default_dtype(torch.float32)

# ~~~~~~~~~~~~~~~~~~~~~~~ General  Functions ~~~~~~~~~~~~~~~~~~~~~~~ #

def save_model(model, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(model, f)
        print("Saved model.")


def load_pca_model():
    with gzip.open('../models/pca_model.pkl.gz', 'rb') as f:
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
    model = ReviewNetBERT(dropout_rate=0.2)
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

    torch.save(model.state_dict(), f'../models/best_model.pth') # large normalized

    train_nn(model, optimizer, file_list, start_epoch, patience=10)


# Modified/made more complex for Model 2+
def train_nn(model, optimizer, file_list, start_epoch=0, batch_size=64, num_epochs=300, patience=5):
    criterion = ordinal_loss
    model.to('cuda')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_loss = float('inf')
    patience_counter = 0
    best_model = None

    predictor_count = 100

    data = None
    for filename in file_list:
        file_data = csv_file_to_nparray(filename)
        if data is None:
            data = file_data
        else:
            data = np.concatenate((np.array(data), np.array(file_data)), axis=0)

    pca = load_pca_model()
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        n_iter = 0

        np.random.shuffle(data)

        pbar = tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i in pbar:
            batch = data[i:i+batch_size]
            inputs = batch[:, :100]
            if type(model) == type(ReviewNetBERT()):
                inputs = convert_to_cuda_tensor(pca_inverse_transform(pca, inputs))
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
        torch.save(best_model, f'../models/best_model_nn5.pth')


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
    train_files = ['train_embeddings_0.csv.gz',
                   'train_embeddings_2.csv.gz',
                   'train_embeddings_1.csv.gz',
                   'train_embeddings_3.csv.gz',
                   'train_embeddings_4.csv.gz',
                   'train_embeddings_5.csv.gz'
    ]
    test_files = ['test_embeddings_0.csv.gz', 'test_embeddings_1.csv.gz']
    train_data = load_and_concatenate_csvs(train_files)
    test_data = load_and_concatenate_csvs(test_files)

    # 1st experiment
    # model = lukasz_train(train_data, test_data)
    # save_model(model, "../models/NB_PCA100.pkl.gz")

    # 2nd experiment, same NB (probabilistic) model with chi-square feature selection for each target
    # lukasz_chi_square_feat_select_experiment_two(train_data, test_data)

    # 3rd experiment
    lukasz_experiment_three_star_rating_ablation_study(train_data, test_data)


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

def lukasz_chi_square_feat_select_experiment_two(train_data: pd.DataFrame, test_data: pd.DataFrame):
    # foo = load_pca_model()
    target_names = ['stars', 'useful', 'funny', 'cool']
    chi2_features = {}

    # Split features and targets
    train_data_x = train_data.drop(target_names, axis=1).values
    test_data_x = test_data.drop(target_names, axis=1).values
    train_data_y = train_data[target_names]
    test_data_y = test_data[target_names]

    # Transform features to the range [0, 1]
    scaler = MinMaxScaler()
    train_data_x_scaled = scaler.fit_transform(train_data_x)
    test_data_x_scaled = scaler.transform(test_data_x)

    selected_features = {}
    models = {}

    print("Performing Chi-Square Feature Selection and Training Models...\n")
    # Perform feature selection and train a model for each target
    for target in target_names:
        # Perform Chi-Square feature selection for the current target
        selector = SelectKBest(score_func=chi2, k=10)
        selector.fit(train_data_x_scaled, train_data_y[target])
        selected_indices = selector.get_support(indices=True)
        selected_features[target] = selected_indices

        # Select features for the current target
        train_x_selected = train_data_x_scaled[:, selected_indices]
        test_x_selected = test_data_x_scaled[:, selected_indices]

        # Train GaussianNB for the current target
        model = CustomGaussianNB()
        model.fit(train_x_selected, train_data_y[target])
        models[target] = model

        # Predict and evaluate
        y_pred = model.predict(test_x_selected)
        print(f"Classification Report for Target: {target}")
        print(classification_report(test_data_y[target], y_pred))
        print("-" * 50)

        save_model(model, f"../models/NB_CHI_{target}_PCA100.pkl.gz")

    # for target in target_names:
    #     train_data_y = train_data[target].values
    #     train_data_x = train_data.drop(target, axis=1).values
    #
    #     # Transform features to the range [0, 1]
    #     scaler = MinMaxScaler()
    #     train_data_x_scaled = scaler.fit_transform(train_data_x)
    #     # test_data_x_scaled = scaler.transform(test_data_x)
    #
    #     # batch_size = 1000  # Number of features to process at a time
    #     # scores = []
    #     # for start in range(0, train_data_x_scaled.shape[1], batch_size):
    #     #     end = start + batch_size
    #     #     x_batch = train_data_x_scaled[:, start:end]
    #     #     chi2_scores, _ = chi2(x_batch, train_data_y)
    #     #     scores.extend(chi2_scores)
    #
    #     chi2_selector = SelectKBest(chi2, k=10)
    #     chi2_selector.fit(train_data_x_scaled, train_data_y)
    #
    #     # Save selected feature indices
    #     chi2_features[target] = chi2_selector.get_support(indices=True)
    #     print(f"Selected features for {target}: {chi2_features[target]}")
    #
    #     test_data_y = test_data[target].values
    #     test_data_x = test_data.drop(target, axis=1).values
    #
    # results = {}
    #
    # for target_train in target_names:
    #     selected_features = chi2_features[target_train]
    #     x_train_selected = train_data.iloc[:, selected_features]
    #
    #     for target_test in target_names:
    #         if target_train == target_test:
    #             continue
    #
    #         # Train on selected features for one target
    #         y_train = train_data[target_train]
    #         model = CustomGaussianNB()
    #         model.fit(x_train_selected, y_train)
    #
    #         # Test on other target
    #         y_test = test_data[target_test]
    #         x_test_selected = test_data.iloc[:, selected_features]
    #         y_pred = model.predict(x_test_selected)
    #
    #         # Evaluate and save results
    #         report = classification_report(y_test, y_pred, output_dict=True)
    #         results[(target_train, target_test)] = report
    #         print(f"Trained on {target_train}, evaluated on {target_test}:")
    #         print(classification_report(y_test, y_pred))


def lukasz_experiment_three_star_rating_ablation_study(train_data: pd.DataFrame, test_data: pd.DataFrame):
    for index in range(1, 6):
        filtered_train_data = train_data[train_data['stars'] != index]
        filtered_test_data = test_data[test_data['stars'] != index]
        model = lukasz_train(filtered_train_data, filtered_test_data)
        save_model(model, f'../models/NB_no_{index}_stars_PCA100.pkl.gz')


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

    NAME = 'J'

    match NAME:
        case 'J':
            joe_main()
        case 'K':
            kyle_main()
        case 'L':
            lukasz_main()
        case _:
            pass