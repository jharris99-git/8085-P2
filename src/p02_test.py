# use *validation* not test here
import numpy as np
import torch
import gzip
import pickle
import SVM
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch import optim, nn

from p01_train import convert_to_cuda_tensor, ReviewNet, csv_file_to_nparray, load_and_concatenate_csvs
from p01_train import CustomGaussianNB


torch.set_default_dtype(torch.float32)
# ~~~~~~~~~~~~~~~~~~~~~~~ General  Functions ~~~~~~~~~~~~~~~~~~~~~~~ #

def load_model(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


def find_robust_max_discrepancy(true_others, pred_others, percentile=99):
    abs_diff = np.abs(true_others - pred_others)
    total_diff_per_sample = np.sum(abs_diff, axis=1)
    robust_max = np.percentile(total_diff_per_sample, percentile)
    return robust_max


def robust_log_normalized_mae(true_others, pred_others, mae, percentile=99):
    robust_max = find_robust_max_discrepancy(true_others, pred_others, percentile)
    normalized_mae = np.log1p(mae) / np.log1p(robust_max)
    return 1 - normalized_mae


def evaluate_model(true_values, predicted_values, continuous, weights=[1, 3]):
    # Separate stars from other metrics
    true_stars = true_values[:, 0]
    pred_stars = predicted_values[:, 0]
    true_others = true_values[:, 1:]
    pred_others = predicted_values[:, 1:]

    mae = mean_absolute_error(true_others, pred_others)
    mae_score = robust_log_normalized_mae(true_others, pred_others, mae)

    if continuous:
        stars_mae = mean_absolute_error(true_stars, pred_stars)

        max_star_error = 5
        normalized_stars_mae = stars_mae / max_star_error
        stars_mae_score = 1 - normalized_stars_mae

        combined_score = (weights[0] * stars_mae_score + weights[1] * mae_score) / sum(weights)

        return {
            'continuous': continuous,
            'stars_mae_score': stars_mae_score,
            'mae_score': mae_score,
            'combined_score': combined_score
        }

    else:
        true_stars = true_stars.astype(int)
        pred_stars = np.round(pred_stars).astype(int)

        stars_accuracy = accuracy_score(true_stars, pred_stars)
        stars_f1 = f1_score(true_stars, pred_stars, average='weighted')

        combined_score = (weights[0] * ((stars_accuracy + stars_f1) / 2) + weights[1] * mae_score) / sum(weights)

        return {
            'continuous': continuous,
            'stars_accuracy': stars_accuracy,
            'stars_f1': stars_f1,
            'mae_score': mae_score,
            'combined_score': combined_score
        }


# ~~~~~~~~~~~~~~~~~~~~~~~~ Joe's  Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #

def test_model(model, test_data, if_cuda):
    model.eval()
    with torch.no_grad():
        inputs = convert_to_cuda_tensor(test_data[:, :100])
        outputs = model(inputs)
        if if_cuda:
            return outputs.cpu().numpy()
        else:
            return outputs.numpy()


def joe_main():
    model = ReviewNet()

    file_list = [
        'test_embeddings_0.csv.gz',
        'test_embeddings_1.csv.gz'
    ]

    model.load_state_dict(torch.load(f'../models/best_model.pth', weights_only=True))
    model.eval()
    model.to('cuda')
    tensor = None
    for filename in file_list:
        file_tensor = csv_file_to_nparray(filename)
        if tensor is None:
            tensor = file_tensor
        else:
            tensor = np.concatenate((np.array(tensor), np.array(file_tensor)), axis=0)
    with torch.no_grad():
        inputs = convert_to_cuda_tensor(tensor[:, :100]).float()
        targets = convert_to_cuda_tensor(tensor[:, 100:]).float()
        outputs = model(inputs)

        true_values = targets.cpu().numpy()
        pred_values = outputs.cpu().numpy()

        evaluation_results = evaluate_model(true_values, pred_values, True)

        print("Evaluation Results:")
        for metric, value in evaluation_results.items():
            if metric != 'continuous':
                print(f"{metric}: {value:.4f}")

        # Check if the combined score is greater than 0.5
        if evaluation_results['combined_score'] > 0.5:
            print("Model performance is satisfactory (score > 0.5)")
        else:
            print("Model performance needs improvement (score <= 0.5)")



# ~~~~~~~~~~~~~~~~~~~~~~~~ Kyle's  Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #

def kyle_main():
    model = load_model("../models/SVM_PCA100.pkl.gz")

    file_list = [
        'test_embeddings_0.csv.gz',
        'test_embeddings_1.csv.gz'
    ]
    data = load_and_concatenate_csvs(file_list)
    test_data_y = data[['stars', 'useful', 'funny', 'cool']].values
    test_data_x = data.drop(['stars', 'useful', 'funny', 'cool'], axis=1)
    pred_y = model.predict(test_data_x)
    evaluation_results = evaluate_model(test_data_y, pred_y, True)
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        if metric != 'continuous':
            print(f"{metric}: {value:.4f}")

    #Trip Advisor
    # data = load_and_concatenate_csvs(['tripadvisor_hotel_reviews_embeddings_0.csv.gz'])
    # test_data_y = data[['stars']].values
    # test_data_x = data.drop(['stars'], axis=1)
    # pred_y = model.predict(test_data_x)
    #
    # true_stars = test_data_y[:, 0]
    # pred_stars = pred_y[:, 0]
    #
    # true_stars = true_stars.astype(int)
    # pred_stars = np.round(pred_stars).astype(int)
    #
    # stars_mae = mean_absolute_error(true_stars, pred_stars)
    #
    # max_star_error = 5
    # normalized_stars_mae = stars_mae / max_star_error
    # stars_mae_score = 1 - normalized_stars_mae
    #
    # evaluation_results = {
    #     'stars_mae_score': stars_mae_score
    # }
    #
    # print("Evaluation Results:")
    # for metric, value in evaluation_results.items():
    #     if metric != 'continuous':
    #         print(f"{metric}: {value:.4f}")
    #
    # stars_accuracy = accuracy_score(true_stars, pred_stars)
    # stars_f1 = f1_score(true_stars, pred_stars, average='weighted')
    #
    # evaluation_results =  {
    #     'stars_accuracy': stars_accuracy,
    #     'stars_f1': stars_f1,
    # }
    #
    # print("Evaluation Results:")
    # for metric, value in evaluation_results.items():
    #     if metric != 'continuous':
    #         print(f"{metric}: {value:.4f}")

# ~~~~~~~~~~~~~~~~~~~~~~~~ Luke's Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #

def lukasz_main():
    file_list = [
        'test_embeddings_0.csv.gz',
        'test_embeddings_1.csv.gz'
    ]
    data = load_and_concatenate_csvs(file_list)

    # lukasz_test_experiment_three(data)

    # model = load_model("../models/NB_no_2_stars_PCA100.pkl.gz")
    #
    # test_data_y = data[['stars', 'useful', 'funny', 'cool']].values
    # test_data_x = data.drop(['stars', 'useful', 'funny', 'cool'], axis=1)
    #
    # pred_y = model.predict(test_data_x)
    # evaluation_results = evaluate_model(test_data_y, pred_y, True)
    #
    # print("Evaluation Results:")
    # for metric, value in evaluation_results.items():
    #     if metric != 'continuous':
    #         print(f"{metric}: {value:.4f}")
    #
    # # Check if the combined score is greater than 0.5
    # if evaluation_results['combined_score'] > 0.5:
    #     print("Model performance is satisfactory (score > 0.5)")
    # else:
    #     print("Model performance needs improvement (score <= 0.5)")

    do_feature_selections(data)


def do_feature_selections(data):
    feat_select_model_files = [
        '../models/NB_CHI_stars_PCA100.pkl.gz',
        '../models/NB_CHI_useful_PCA100.pkl.gz',
        '../models/NB_CHI_funny_PCA100.pkl.gz',
        '../models/NB_CHI_cool_PCA100.pkl.gz'
    ]

    models = []
    for file in feat_select_model_files:
        models.append(load_model(file))

    feats = [  # stars, useful, funny, cool selected feature indices
        [0, 1, 2, 3, 5, 7, 10, 11, 12, 17],
        [0, 1, 2, 3, 4, 5, 6, 19, 21, 22],
        [0, 1, 3, 4, 6, 8, 13, 19, 21, 33],
        [0, 1, 2, 4, 6, 10, 19, 21, 22, 39]
    ]
    targets = ['stars', 'useful', 'funny', 'cool']

    all_pred_y = []
    all_test_data_y = []

    for index, target in enumerate(targets):
        test_data_y = data[targets].values
        feature_indices = feats[index]
        test_data_x = data.drop(targets, axis=1)
        test_data_x = test_data_x.iloc[:, feature_indices]

        pred_y = models[index].predict(test_data_x)

        all_pred_y.append(pred_y.reshape(-1, 1))  # Ensure 2D shape for stacking
        all_test_data_y.append(test_data_y[:, index].reshape(-1, 1))

    # Combine all predictions and true values along the column axis
    combined_pred_y = np.column_stack(all_pred_y)
    combined_test_data_y = np.column_stack(all_test_data_y)

    evaluation_results = evaluate_model(combined_test_data_y, combined_pred_y, True)

    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        if metric != 'continuous':
            print(f"{metric}: {value:.4f}")

    # Check if the combined score is greater than 0.5
    if evaluation_results['combined_score'] > 0.5:
        print("Model performance is satisfactory (score > 0.5)")
    else:
        print("Model performance needs improvement (score <= 0.5)")


def lukasz_test_experiment_three(data):
    models = [
        load_model("../models/NB_no_1_stars_PCA100.pkl.gz"),
        load_model("../models/NB_no_2_stars_PCA100.pkl.gz"),
        load_model("../models/NB_no_3_stars_PCA100.pkl.gz"),
        load_model("../models/NB_no_4_stars_PCA100.pkl.gz"),
        load_model("../models/NB_no_5_stars_PCA100.pkl.gz")
    ]

    test_data_y = data[['stars', 'useful', 'funny', 'cool']].values
    test_data_x = data.drop(['stars', 'useful', 'funny', 'cool'], axis=1)

    index = 1
    for model in models:
        pred_y = model.predict(test_data_x)
        evaluation_results = evaluate_model(test_data_y, pred_y, True)

        print(f"Evaluation Results (model trained without {index} star ratings):")
        for metric, value in evaluation_results.items():
            if metric != 'continuous':
                print(f"{metric}: {value:.4f}")

        # Check if the combined score is greater than 0.5
        if evaluation_results['combined_score'] > 0.5:
            print("Model performance is satisfactory (score > 0.5)")
        else:
            print("Model performance needs improvement (score <= 0.5)")

        index += 1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
if __name__ == '__main__':

    NAME = 'K'
    match NAME:
        case 'J':
            joe_main()
        case 'K':
            kyle_main()
            pass
        case 'L':
            lukasz_main()
        case _:
            pass