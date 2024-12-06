import gzip
import os
import pickle
from math import ceil

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from transformers import BertTokenizer, BertModel

DEBUG = False  # whether to use tweak or train for prepro

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 5000)

np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: "%.4g" % x))

BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_model = BERT_model.to(device)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PREPRO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def encode_text(texts, max_length=128):
    encoded = BERT_tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return {k: v.to(device) for k, v in encoded.items()}


def get_bert_embeddings(encoded_text):
    with torch.no_grad():
        outputs = BERT_model(**encoded_text)
    embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings_cpu = embeddings.cpu().numpy()
    torch.cuda.empty_cache()
    return embeddings_cpu


def preprocess_dataset(texts, chunk, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size].tolist()
        encoded = encode_text(batch)
        embedding = get_bert_embeddings(encoded)
        embeddings.append(embedding)
        torch.cuda.empty_cache()
        print("Chunk ", chunk, " Batch ", i // batch_size + 1, " of ", ceil(len(texts) / batch_size), " complete.", sep='')
    return np.vstack(embeddings)


def load_or_initialize_pca(n_components=100):
    pca_file_path = '../models/pca_model.pkl.gz'
    if os.path.exists(pca_file_path):
        with gzip.open(pca_file_path, 'rb') as f:
            pca = pickle.load(f)
            print("Loaded existing PCA model.")
    else:
        pca = IncrementalPCA(n_components=n_components)
        print("Initialized new PCA model.")
    return pca


def save_pca_model(pca):
    with gzip.open('../models/pca_model.pkl.gz', 'wb') as f:
        pickle.dump(pca, f)
        print("Saved PCA model.")


def process_and_fit_pca(dataset_url):
    # Initialize PCA

    current_chunk = 0

    pca = load_or_initialize_pca()

    # Process each chunk to fit PCA
    for chunk in pd.read_csv(dataset_url, sep='|', chunksize=10000):  # Read in smaller chunks
        embeddings = preprocess_dataset(chunk['text'], current_chunk)  # Process current chunk

        # Incrementally fit PCA on new embeddings
        pca.partial_fit(embeddings)

        current_chunk += 1

    # Save the updated PCA model after processing all chunks
    save_pca_model(pca)


def process_and_transform_embeddings(dataset_url, output_file_base):
    # Load the trained PCA model
    pca = load_or_initialize_pca()

    # Prepare to save reduced embeddings
    output_file_index = 0
    output_file = f"{output_file_base}_{output_file_index}.csv"

    # Create a directory for outputs if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_rows = 0

    current_chunk = 0

    # Process each chunk to transform embeddings
    for chunk in pd.read_csv(dataset_url, sep='|', chunksize=10000):  # Read in smaller chunks
        embeddings = preprocess_dataset(chunk['text'], current_chunk)  # Process current chunk

        # Transform embeddings using the fitted PCA model
        reduced_embeddings = pca.transform(embeddings)

        current_chunk += 1

        # Prepare DataFrame for saving reduced embeddings
        prepro_dataset = pd.DataFrame(reduced_embeddings)

        # Append stars, useful, funny, cool columns from original dataset
        prepro_dataset[['stars', 'useful', 'funny', 'cool']] = chunk[['stars', 'useful', 'funny', 'cool']].reset_index(
            drop=True)

        # Save in batches of 1 million lines
        for start in range(0, len(prepro_dataset), 1000000):
            end = min(start + 1000000, len(prepro_dataset))
            chunk_to_save = prepro_dataset.iloc[start:end]

            if total_rows == 0:
                chunk_to_save.to_csv(output_file, sep='|', index=False)
            else:
                chunk_to_save.to_csv(output_file, sep='|', index=False, header=False, mode='a')

            total_rows += len(chunk_to_save)

            # If we reach 1M rows written, move to next output file
            if total_rows >= 1000000:
                output_file_index += 1
                output_file = f"{output_file_base}_{output_file_index}.csv"
                total_rows = 0  # Reset line count for the new file


def format_and_compress_files(file_list, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for file_name in sorted(file_list):  # Sort to ensure sequential processing
        input_file_path = '../datasets/' + file_name
        output_file_path = output_directory + f"{file_name}.gz"

        with open(input_file_path, 'r') as infile:
            # Read all lines from the input file
            lines = infile.readlines()

        processed_lines = []
        for line in lines:
            # Split line into fields (assuming whitespace or comma-separated)
            fields = line.strip().split('|')
            # Format each field to 7 decimal places if it's a float
            formatted_fields = [f"{float(field):.7f}" if is_float(field) else field for field in fields]
            processed_lines.append('|'.join(formatted_fields))

        # Write processed lines to a gzip-compressed file
        with gzip.open(output_file_path, 'wt') as outfile:
            outfile.write('\n'.join(processed_lines))


def is_float(value):
    try:
        float(value)
        if '.' in value:
            return True
        return False
    except ValueError:
        return False


if __name__ == '__main__':
    files = ['tweak_embeddings_0.csv']

    output_directory = '../datasets/'

    format_and_compress_files(files, output_directory)

    # base_url = '../datasets/train' if not DEBUG else '../datasets/tweak'
    # dataset_url = base_url + '.csv'

    # process_and_fit_pca(dataset_url)

    # if not DEBUG:
        # base_url = '../datasets/test'
        # dataset_url = base_url + '.csv'

        # process_and_fit_pca(dataset_url)


        # base_url = '../datasets/val'
        # dataset_url = base_url + '.csv'

        # process_and_fit_pca(dataset_url)

        # base_url = '../datasets/train'
        # dataset_url = base_url + '.csv'

        # Define output file base name
        # output_file_base = base_url + '_embeddings'

        # process_and_transform_embeddings(dataset_url, output_file_base)

        # base_url = '../datasets/test'
        # dataset_url = base_url + '.csv'

        # Define output file base name
        # output_file_base = base_url + '_embeddings'

        # process_and_transform_embeddings(dataset_url, output_file_base)

        # base_url = '../datasets/val'
        # dataset_url = base_url + '.csv'

        # Define output file base name
        # output_file_base = base_url + '_embeddings'

        # process_and_transform_embeddings(dataset_url, output_file_base)