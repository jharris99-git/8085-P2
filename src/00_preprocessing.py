import numpy as np
import torch

import re

import nltk
import pandas as pd
import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel

DEBUG = True # whether to use tweak or train for prepro
EXPERIMENT = False

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 5000)

nltk.download('punkt')
nltk.download('stopwords')

spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_model = BertModel.from_pretrained('bert-base-uncased')


# ~~~~~~~~~~~~~~~~~~~~~~~ PREPRO EXPERIMENTS ~~~~~~~~~~~~~~~~~~~~~~~ #

def basic_preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)


def fast_preprocess(text_series):
    return (text_series.str.lower()
                       .str.replace('[^\w\s]\s*', '', regex=True)
                       .str.split()
                       .str.join(' '))


def spacy_preprocess(text):
    doc = spacy_nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])


def BERT_minimal_preprocess(text):
    return ' '.join(BERT_tokenizer.tokenize(text))


def create_tfidf_vectorizer(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(texts)
    return vectorizer


def extract_features(texts, vectorizer):
    return vectorizer.transform(texts)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PREPRO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def encode_text(text, max_length=512):
    encoded = BERT_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoded


def get_bert_embeddings(encoded_text):
    with torch.no_grad():
        outputs = BERT_model(**encoded_text)
    # Use the [CLS] token embedding (first token)
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
    return embeddings.numpy()


def preprocess_dataset(texts):
    embeddings = []
    for text in texts:
        encoded = encode_text(text)
        embedding = get_bert_embeddings(encoded)
        embeddings.append(embedding)
    return np.array(embeddings)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == '__main__':

    dataset_url = '../datasets/train.csv' if not DEBUG else '../datasets/tweak.csv'

    #
    if EXPERIMENT: #  TODO: Figure out a way to do the preprocess_dataset over sections with consistent results
        chunk_list = []
        for chunk in pd.read_csv(dataset_url, chunksize=10000, sep='|'):
            chunk['processed_text'] = chunk['text'].apply(BERT_minimal_preprocess)
            # chunk['processed_text'] = fast_preprocess(chunk['text'])

            chunk_list.append(chunk)
        full_df = pd.concat(chunk_list, ignore_index=True)
        print(full_df.head(10))
    else:
        dataset = pd.read_csv(dataset_url, sep='|')
        X = preprocess_dataset(dataset.text)
        print(X)

