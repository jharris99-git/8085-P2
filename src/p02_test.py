# use *validation* not test here
import torch
from torch import optim, nn

from p01_train import convert_to_cuda_tensor, ReviewNet, csv_file_to_nparray


# ~~~~~~~~~~~~~~~~~~~~~~~~ Joe's  Functions ~~~~~~~~~~~~~~~~~~~~~~~~ #


def joe_main():
    model = ReviewNet()
    optimizer = optim.Adam(model.parameters())

    file_list = [
        'test_embeddings_0.csv.gz' #,
        # 'test_embeddings_1.csv.gz'
    ]

    model.load_state_dict(torch.load(f'../models/model_50.pth'))
    model.eval()
    model.to('cuda')

    tensor = csv_file_to_nparray(f'test_embeddings_0.csv.gz')
    with torch.no_grad():
        inputs = convert_to_cuda_tensor(tensor[:, :100]).float()
        targets = convert_to_cuda_tensor(tensor[:, 100:]).float()
        outputs = model(inputs)
        rounded_outputs = torch.round(outputs).int()
        # loss = nn.MSELoss(outputs, targets)
        # print(outputs)
        # print("Loss: ", loss, sep='')
        comp = zip(targets, rounded_outputs)
        for true, pred in comp:
            print(f'True  Stars: {true[0].item():.2f}  Pred. Stars: {pred[0].item():.2f}  Loss: {(true[0].item() - pred[0].item())**2:.4f}\n'
                  f'True  Useful: {true[1].item():.2f} Pred. Useful: {pred[1].item():.2f} Loss: {(true[1].item() - pred[1].item())**2:.4f}\n'
                  f'True  Funny: {true[2].item():.2f}  Pred. Funny: {pred[2].item():.2f}  Loss: {(true[2].item() - pred[2].item())**2:.4f}\n'
                  f'True  Cool: {true[3].item():.2f}   Pred. Cool: {pred[3].item():.2f}   Loss: {(true[3].item() - pred[3].item())**2:.4f}\n')



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