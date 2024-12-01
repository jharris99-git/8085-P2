# use *validation* not test here
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch import optim, nn

from p01_train import convert_to_cuda_tensor, ReviewNet, csv_file_to_nparray


def evaluate_model(true_values, predicted_values, continuous):
    # Separate stars from other metrics
    true_stars = true_values[:, 0]
    pred_stars = predicted_values[:, 0]
    true_others = true_values[:, 1:]
    pred_others = predicted_values[:, 1:]

    mae = mean_absolute_error(true_others, pred_others)
    rmse = np.sqrt(mean_squared_error(true_others, pred_others))

    if continuous:
        stars_mse = mean_squared_error(true_stars, pred_stars)
        stars_mae = mean_absolute_error(true_stars, pred_stars)

        combined_score = ((1 - stars_mae / 5) + (1 - stars_mse / 5) + (1 - mae / 5) + (1 - rmse / 5)) / 4

        return {
            'continuous': continuous,
            'stars_mae': stars_mae,
            'stars_mse': stars_mse,
            'other_mae': mae,
            'other_rmse': rmse,
            'combined_score': combined_score
        }

    else:
        true_stars = true_stars.astype(int)
        pred_stars = np.round(pred_stars).astype(int)

        stars_accuracy = accuracy_score(true_stars, pred_stars)
        stars_f1 = f1_score(true_stars, pred_stars, average='weighted')

        combined_score = (stars_accuracy + stars_f1 + (1 - mae / 5) + (1 - rmse / 5)) / 4

        return {
            'continuous': continuous,
            'stars_accuracy': stars_accuracy,
            'stars_f1': stars_f1,
            'other_mae': mae,
            'other_rmse': rmse,
            'combined_score': combined_score
        }

    # Compute metrics for other outputs


    # Combine into a single score (you can adjust weights as needed)
    #




# ~~~~~~~~~~~~~~~~~~~~~~~~ Joe's  Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #


def joe_main():
    model = ReviewNet()
    optimizer = optim.Adam(model.parameters())

    file_list = [
        'test_embeddings_0.csv.gz',
        'test_embeddings_1.csv.gz'
    ]

    model.load_state_dict(torch.load(f'../models/best_model.pth'))
    model.eval()
    model.to('cuda')

    tensor = csv_file_to_nparray(f'test_embeddings_0.csv.gz')
    with torch.no_grad():
        inputs = convert_to_cuda_tensor(tensor[:, :100]).float()
        targets = convert_to_cuda_tensor(tensor[:, 100:]).float()
        outputs = model(inputs)
        rounded_outputs = torch.round(outputs).int()

        true_values = targets.cpu().numpy()
        pred_values = outputs.cpu().numpy() # or # rounded_outputs.cpu().numpy()

        evaluation_results = evaluate_model(true_values, pred_values, True)

        print("Evaluation Results:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value:.4f}")

        # Check if the combined score is greater than 0.5
        if evaluation_results['combined_score'] > 0.5:
            print("Model performance is satisfactory (score > 0.5)")
        else:
            print("Model performance needs improvement (score <= 0.5)")

        # comp = zip(targets, rounded_outputs)
        # for true, pred in comp:
        #     print(f'True  Stars: {true[0].item():.2f}  Pred. Stars: {pred[0].item():.2f}  Loss: {(true[0].item() - pred[0].item())**2:.4f}\n'
        #           f'True  Useful: {true[1].item():.2f} Pred. Useful: {pred[1].item():.2f} Loss: {(true[1].item() - pred[1].item())**2:.4f}\n'
        #           f'True  Funny: {true[2].item():.2f}  Pred. Funny: {pred[2].item():.2f}  Loss: {(true[2].item() - pred[2].item())**2:.4f}\n'
        #           f'True  Cool: {true[3].item():.2f}   Pred. Cool: {pred[3].item():.2f}   Loss: {(true[3].item() - pred[3].item())**2:.4f}\n')



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