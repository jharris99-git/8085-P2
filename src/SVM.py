import pandas as pd
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC, LinearSVC


def train_model(train_data: pd.DataFrame):

    train_data_y = train_data[['stars', 'useful', 'funny', 'cool']].values
    train_data_x = train_data.drop(['stars', 'useful', 'funny', 'cool'], axis=1).values

    model = MultiOutputClassifier(LinearSVC(verbose=1), n_jobs=-1)
    # model = MultiOutputClassifier(SGDClassifier(loss='hinge', verbose=1, n_jobs=-1), n_jobs=-1)
    model.fit(train_data_x,train_data_y)

    return model

def use_model(model, test_data: pd.DataFrame):
    test_data_y = test_data[['stars', 'useful', 'funny', 'cool']].values
    test_data_x = test_data.drop(['stars', 'useful', 'funny', 'cool'], axis=1).values

    # Predict using the input data rows.
    y_pred = model.predict(test_data_x)

    print("\nHere is the classification report:")
    for true, pred in zip(test_data_y.T, y_pred.T):
        print(set(true) - set(pred))
        print(classification_report(y_true=true, y_pred=pred))