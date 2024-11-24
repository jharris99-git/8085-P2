# define training code
import gzip
import pickle


def load_pca_model():
    with gzip.open('../models/PCA.pkl.gz', 'rb') as f:
        return pickle.load(f)

def pca_inverse_transform(pca, pca_100):
    return pca.inverse_transform(pca_100)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
if __name__ == '__main__':

    # base_data = pd.read_csv('../datasets/', low_memory=False)

    # base_data = process_data(base_data) # preprocess

    NAME = 'K'

    match NAME:
        case 'J':
            pass
        case 'K':
            pass
        case 'L':
            pass
        case _:
            pass