import torch

from NeuralNets import ReviewNetBERT
from p01_train import csv_file_to_nparray, load_and_concatenate_csvs, CustomGaussianNB # prevent false 'unused import' flag
from p02_test import test_model, evaluate_model, load_model

torch.set_default_dtype(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print(f'      Validate Models      \n'
          f'~~~~~~~~~~=======~~~~~~~~~~\n')

    file = 'val_embeddings_0.csv.gz'

    while True:
        print(f'Available models:\n'
              f'1. Neural Network\n'
              f'2. Naive-Bayes\n'
              f'3. SVM\n')
        choice = str(input('Select the model you wish to test: (1/2/3) ')).lower().strip()

        if choice in ['1', 'Neural Network', '1.', 'nn']:
            cont_choice = input('Would you like Stars to be evaluated as (a) discrete or (b) continuous between [0,5]: (a,b) ')
            cont = True if cont_choice == 'b' else False
            tensor = csv_file_to_nparray(file)

            model = ReviewNetBERT()
            model.load_state_dict(torch.load(f'../models/NN_BERT768.pth', weights_only=True))
            model.eval()

            is_cuda = True if torch.cuda.is_available() else False
            if is_cuda:
                model.to('cuda')

            true = tensor[:, 100:]
            pred = test_model(model, tensor, is_cuda)

            evaluation_results = evaluate_model(true, pred, cont)

            print("Evaluation Results:")
            for metric, value in evaluation_results.items():
                if metric != 'continuous':
                    print(f"{metric}: {value:.4f}")

        if choice in ['2', 'Naive-Bayes', '2.', 'nb']:
            cont_choice = input(
                'Would you like Stars to be evaluated as (a) discrete or (b) continuous between [0,5]: (a,b)')
            cont = True if cont_choice == 'b' else False
            data = load_and_concatenate_csvs([file])
            test_data_y = data[['stars', 'useful', 'funny', 'cool']].values
            test_data_x = data.drop(['stars', 'useful', 'funny', 'cool'], axis=1)

            model = load_model("../models/NB_PCA100.pkl.gz")
            pred_y = model.predict(test_data_x)
            evaluation_results = evaluate_model(test_data_y, pred_y, cont)

            print("Evaluation Results:")
            for metric, value in evaluation_results.items():
                if metric != 'continuous':
                    print(f"{metric}: {value:.4f}")

        if choice in ['3', 'SVM', '3.', 'Support Vector Machine']:
            cont_choice = input(
                'Would you like Stars to be evaluated as (a) discrete or (b) continuous between [0,5]: (a,b)')
            cont = True if cont_choice == 'b' else False

            data = load_and_concatenate_csvs([file])
            test_data_y = data[['stars', 'useful', 'funny', 'cool']].values
            test_data_x = data.drop(['stars', 'useful', 'funny', 'cool'], axis=1)

            model = load_model("../models/SVM_PCA100.pkl.gz")
            pred_y = model.predict(test_data_x)
            evaluation_results = evaluate_model(test_data_y, pred_y, cont)

            print("Evaluation Results:")
            for metric, value in evaluation_results.items():
                if metric != 'continuous':
                    print(f"{metric}: {value:.4f}")

        choice = str(input('\n\nWould you like to continue? (Y/N) ')).lower().strip()
        if choice in ['n', 'no', '0']:
            break
