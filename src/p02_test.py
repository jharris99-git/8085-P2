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
        # loss = nn.MSELoss(outputs, targets)
        print(outputs)
        # print("Loss: ", loss, sep='')


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