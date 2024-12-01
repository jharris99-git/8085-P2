# use *validation* not test here
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch import optim, nn

from p01_train import convert_to_cuda_tensor, ReviewNet, csv_file_to_nparray

torch.set_default_dtype(torch.float32)

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
            'stars_mae_score': stars_mae,
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

    model.load_state_dict(torch.load(f'../models/best_model.pth'))
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
if __name__ == '__main__':


    # base_data = pd.read_csv('../datasets/', low_memory=False)

    # base_data = process_data(base_data) # preprocess

    NAME = 'J'

    match NAME:
        case 'J':
            joe_main()
        case 'K':
            # kyle_main()
            pass
        case 'L':
            pass
        case _:
            pass