from scipy.io import arff

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, fbeta_score

import numpy as np


def load_data(file_path):
    data = arff.loadarff('data/a.arff')
    df = pd.DataFrame(data[0])
    df.rename(columns={'class': 'label'}, inplace=True)
    df['label'] = pd.to_numeric(df['label'])
    return df


def decision_tree(train_X, val_X, train_y):
    DT_model = tree.DecisionTreeClassifier()
    DT_model.fit(train_X, train_y)
    val_preds = DT_model.predict(val_X)
    return val_preds


def naive_bayes(train_X, val_X, train_y):
    NB_model = GaussianNB()
    NB_model.fit(train_X, train_y)
    val_preds = NB_model.predict(val_X)
    return val_preds


def kNN(train_X, val_X, train_y, n_neighbors):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(train_X,train_y)
    val_preds = knn_model.predict(val_X)
    return val_preds


def main():
    df = load_data('data/a.arff')
    # split
    # X
    a_features = ['x', 'y']
    X = df[a_features]
    # y
    y = df.label
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # DT
    val_preds = decision_tree(train_X, val_X, train_y)
    f1_score_value = f1_score(val_y, val_preds,
                              average='weighted', labels=np.unique(val_y))
    print('DT fscore: {}'.format(f1_score_value))

    # NB
    val_preds = naive_bayes(train_X, val_X, train_y)
    f1_score_value = f1_score(val_y, val_preds,
                              average='weighted', labels=np.unique(val_y))
    print('NB fscore: {}'.format(f1_score_value))

    # KNN (k = 1)
    val_preds = kNN(train_X, val_X, train_y, 1)
    f1_score_value = f1_score(val_y, val_preds,
                              average='weighted', labels=np.unique(val_y))
    print('KNN(k = 1) fscore: {}'.format(f1_score_value))

    # KNN (k = 21)
    val_preds = kNN(train_X, val_X, train_y, 21)
    np.reshape(val_preds, (len(val_preds), -1))
    f1_score_value = f1_score(val_y, val_preds,
                              average='weighted', labels=np.unique(val_y))
    print('KNN(k = 21) fscore: {}'.format(f1_score_value))


if __name__ == "__main__":
    main()
print('Done!')
