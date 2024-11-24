import numpy as np
import torch
print(torch.cuda.is_available())
# import re

# import nltk
import pandas as pd
# import spacy
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel

DEBUG = True # whether to use tweak or train for prepro
EXPERIMENT_BERT = False

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 5000)

# nltk.download('punkt')
# nltk.download('stopwords')

# spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_model = BERT_model.to(device)


# ~~~~~~~~~~~~~~~~~~~~~~~ PREPRO EXPERIMENTS ~~~~~~~~~~~~~~~~~~~~~~~ #

# def basic_preprocess(text):
#     # Convert to lowercase
#     text = text.lower()
#     # Remove special characters and digits
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     # Tokenize
#     tokens = word_tokenize(text)
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words]
#     return ' '.join(tokens)
#
#
# def fast_preprocess(text_series):
#     return (text_series.str.lower()
#                        .str.replace('[^\w\s]\s*', '', regex=True)
#                        .str.split()
#                        .str.join(' '))
#
#
# def spacy_preprocess(text):
#     doc = spacy_nlp(text)
#     return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])
#
#
# def BERT_minimal_preprocess(text):
#     return ' '.join(BERT_tokenizer.tokenize(text))
#
#
# def create_tfidf_vectorizer(texts, max_features=5000):
#     vectorizer = TfidfVectorizer(max_features=max_features)
#     vectorizer.fit(texts)
#     return vectorizer
#
#
# def extract_features(texts, vectorizer):
#     return vectorizer.transform(texts)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PREPRO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def encode_text(text, max_length=128):
    encoded = BERT_tokenizer.encode_plus(
        text,
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
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
    return embeddings.cpu().numpy()


def preprocess_dataset(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = encode_text(batch)
        embedding = get_bert_embeddings(encoded)
        embeddings.extend(embedding)
    return np.array(embeddings)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == '__main__':

    dataset_url = '../datasets/train.csv' if not DEBUG else '../datasets/tweak.csv'

    dtypes = {
        'stars':'float32',
        'useful':'int16',
        'funny':'int16',
        'cool':'int16',
        'text':str
    }
    dataset = pd.read_csv(dataset_url, sep='|', )
    X = preprocess_dataset(dataset.text)
    print(X)

