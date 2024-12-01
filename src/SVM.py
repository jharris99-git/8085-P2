import pandas as pd

from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

def train_model(train_data: pd.DataFrame, test_data: pd.DataFrame):

    train_data_y = train_data[['stars', 'useful', 'funny', 'cool']].values
    train_data_x = train_data.drop(['stars', 'useful', 'funny', 'cool'], axis=1).values

    test_data_y = train_data[['stars', 'useful', 'funny', 'cool']].values
    test_data_x = train_data.drop(['stars', 'useful', 'funny', 'cool'], axis=1).values

    model = MultiOutputClassifier(SVC(kernel='linear'), n_jobs=-1)
    model.fit(train_data_x,train_data_y)
    pred_y = model.predict(test_data_x)

    print ("\nHere is the classification report:")
    for true, pred in zip(test_data_y.T, pred_y.T):
        print(set(true) - set(pred))
        print (classification_report(y_true=true, y_pred=pred))

    return model