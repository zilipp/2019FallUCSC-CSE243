from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, fbeta_score
import arff
import numpy as np
import graphviz

oneD_arr_to_twoD_arr_func = lambda x: np.reshape(x, (len(x), -1))

def loadData():
    dataset = arff.load(open('data/sick.arff', 'r'))
    data = np.array(dataset['data'])

    # load data into 2d array
    X = oneD_arr_to_twoD_arr_func(data[:, 0]).astype(np.float)
    Y = oneD_arr_to_twoD_arr_func(data[:, 1]).astype(np.float)
    X = np.column_stack((X, Y))
    label = oneD_arr_to_twoD_arr_func(data[:, 2]).astype(int)
    return X, label


def decision_tree(X_train, label_train, X_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, label_train)
    label_predict = clf.predict(X_test)
    label_predict = oneD_arr_to_twoD_arr_func(label_predict)
    return label_predict


def naive_bayes(X_train, label_train, X_test):
    gnb = GaussianNB()
    gnb = gnb.fit(X_train, label_train)
    label_predict = gnb.predict(X_test)
    label_predict = oneD_arr_to_twoD_arr_func(label_predict)
    return label_predict


def kNN(X_train, label_train, X_test, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, label_train)
    label_predict = knn.predict(X_test)
    label_predict = oneD_arr_to_twoD_arr_func(label_predict)
    return label_predict


def main():
    X, label = loadData()
    # split into random training and testing data
    label = label == 1
    # label = np.invert(label)
    skf = StratifiedKFold(n_splits=5)
    for train, test in skf.split(X, label):
        X_train = X[train, :]
        X_test = X[test, :]
        label_train = label[train, :]
        label_test = label[test, :]
        # Decision Tree
        label_predict = decision_tree(X_train, label_train, X_test)
        f1_score_value = f1_score(label_test, label_predict,
                                  average='weighted', labels=np.unique(label_test))
        print('DT fscore: {}'.format(f1_score_value))

        # Gaussian Naive Bayes
        label_predict = naive_bayes(X_train, label_train, X_test)
        f1_score_value = fbeta_score(label_test, label_predict, beta=1.0,
                                     average='weighted', labels=np.unique(label_test))
        print('NB fscore: {}'.format(f1_score_value))

        # KNN k = 21
        k = 21
        label_predict = kNN(X_train, label_train, X_test, n_neighbors=k)
        f1_score_value = fbeta_score(label_test, label_predict, beta=1.0,
                                    average='weighted', labels=np.unique(label_test))
        print('KNN k={} fscore: {}'.format(k, f1_score_value))


# # draw decision tree graph
# # Need run the following command in terminal to install graphviz on macOS: brew install graphviz
# tree.plot_tree(clf.fit(X_train, label_train))
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")
if __name__ == "__main__":
    main()

print('Done!')