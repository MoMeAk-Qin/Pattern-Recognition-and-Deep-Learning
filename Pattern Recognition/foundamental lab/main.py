# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
import pandas as pd
from sklearn import svm
import timeit
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


def load_mnist():
    mnist = fetch_openml('mnist_784', data_home="./dataset", cache=False)
    X, Y = mnist['data'], mnist['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    return X_train, X_test, Y_train, Y_test, X


def knn():
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, Y_train)
    score = knn_clf.score(X_test, Y_test)
    return score


def pca_knn():
    pca = PCA(n_components=100)
    pca.fit(X_train, Y_train)
    X_train_reduction = pca.transform(X_train)
    X_test_reduction = pca.transform(X_test)
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_reduction, Y_train)
    score = knn_clf.score(X_test_reduction, Y_test)
    return score


def pca_mnist_perfor():
    pca = PCA()
    pca.n_components = 784
    pca_data = pca.fit_transform(X)
    percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
    cum_var_explained = np.cumsum(percentage_var_explained)
    x = np.array(range(0, 100))
    y = np.array(percentage_var_explained[:100])
    plt.figure(1, figsize=(6, 4))
    # plt.bar(x, y)
    # plt.xlabel('n_components')
    # plt.ylabel('Contribution rate')
    # plt.show()
    plt.plot(cum_var_explained, linewidth=2)
    plt.axis('tight')
    plt.grid()
    plt.xlabel('n_components')
    plt.ylabel('Cumulative_explained_variance')
    plt.show()


def test_time():
    foo1 = """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.decomposition import PCA
    mnist = fetch_openml('mnist_784', data_home="./dataset")
    X, y = mnist['data'], mnist['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
    knn_clf = KNeighborsClassifier()
    pca = PCA(n_components=60)
    pca.fit(X_train, y_train)
    X_train_reduction = pca.transform(X_train)
    X_test_reduction = pca.transform(X_test)
    knn_clf.fit(X_train_reduction, y_train)
    knn_clf.score(X_test_reduction, y_test)
    """
    foo2 = """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    mnist = fetch_openml('mnist_784', data_home="./dataset")
    X, y = mnist['data'], mnist['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    knn_clf.score(X_test, y_test)
    """
    time_knn = timeit.timeit(stmt=foo2, number=1)
    time_pca = timeit.timeit(stmt=foo1, number=1)
    print(time_knn)
    print(time_pca)


def gmm():
    gmm = GaussianMixture(n_components=3).fit(iris_X)
    print(gmm.get_params(iris_X))
    labels = gmm.predict(iris_X)
    count1 = 0
    count2 = 0
    count3 = 0
    sum2 = np.sum(labels[:50] == 0)
    sum0 = np.sum(labels[:50] == 1)
    sum1 = np.sum(labels[:50] == 2)
    if sum1 > sum2 and sum1 > sum0:
        count1 = sum2 + sum0
    elif sum0 > sum1 and sum0 > sum2:
        count1 = sum1 + sum2
    else:
        count1 = sum1 + sum0
    sum2 = np.sum(labels[50:100] == 0)
    sum0 = np.sum(labels[50:100] == 1)
    sum1 = np.sum(labels[50:100] == 2)
    if sum1 > sum2 and sum1 > sum0:
        count2 = sum2 + sum0
    elif sum0 > sum1 and sum0 > sum2:
        count2 = sum1 + sum2
    else:
        count2 = sum1 + sum0
    sum2 = np.sum(labels[100:150] == 0)
    sum0 = np.sum(labels[100:150] == 1)
    sum1 = np.sum(labels[100:150] == 2)
    if sum1 > sum2 and sum1 > sum0:
        count3 = sum2 + sum0
    elif sum0 > sum1 and sum0 > sum2:
        count3 = sum1 + sum2
    else:
        count3 = sum1 + sum0
    accuracy = 1 - (count1 + count2 + count3) / 150
    return accuracy


def svm_t():
    model = svm.SVC()
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    accuarcy = metrics.accuracy_score(prediction, test_y)
    return accuarcy


def spclus():
    scores = []
    s = dict()
    for index, gamma in enumerate((0.01, 0.1, 1, 10)):
        for index, k in enumerate((3, 4, 5, 6)):
            pred_y = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(iris_X)
            print(
                "Calinski-Harabasz Score with gamma=",
                gamma,
                "n_cluster=",
                k,
                "score=",
                metrics.calinski_harabaz_score(iris_X, pred_y),
            )
            tmp = dict()
            tmp["gamma"] = gamma
            tmp["n_cluster"] = k
            tmp["score"] = metrics.calinski_harabaz_score(iris_X, pred_y)
            s[metrics.calinski_harabaz_score(iris_X, pred_y)] = tmp
            scores.append(metrics.calinski_harabaz_score(iris_X, pred_y))
    print(np.max(scores))
    print("最大得分项：")
    print(s.get(np.max(scores)))


def best_spclu():
    pred_y = SpectralClustering(n_clusters=3, gamma=1).fit_predict(iris_X)
    count1 = 0
    count2 = 0
    count3 = 0
    sum2 = np.sum(pred_y[:50] == 0)
    sum0 = np.sum(pred_y[:50] == 1)
    sum1 = np.sum(pred_y[:50] == 2)
    if sum1 > sum2 and sum1 > sum0:
        count1 = sum2 + sum0
    elif sum0 > sum1 and sum0 > sum2:
        count1 = sum1 + sum2
    else:
        count1 = sum1 + sum0
    sum2 = np.sum(pred_y[50:100] == 0)
    sum0 = np.sum(pred_y[50:100] == 1)
    sum1 = np.sum(pred_y[50:100] == 2)
    if sum1 > sum2 and sum1 > sum0:
        count2 = sum2 + sum0
    elif sum0 > sum1 and sum0 > sum2:
        count2 = sum1 + sum2
    else:
        count2 = sum1 + sum0
    sum2 = np.sum(pred_y[100:150] == 0)
    sum0 = np.sum(pred_y[100:150] == 1)
    sum1 = np.sum(pred_y[100:150] == 2)
    if sum1 > sum2 and sum1 > sum0:
        count3 = sum2 + sum0
    elif sum0 > sum1 and sum0 > sum2:
        count3 = sum1 + sum2
    else:
        count3 = sum1 + sum0
    return 1 - (count1 + count2 + count3) / 150


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test, X = load_mnist()
    iris = pd.read_csv("./dataset/Iris.csv")
    iris.drop("Id", axis=1, inplace=True)
    train, test = train_test_split(iris, test_size=0.3)
    train_X = train[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    train_y = train["Species"]
    test_X = test[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    test_y = test["Species"]
    iris_X = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    iris_Y = iris["Species"]
    accuarcy_knn = knn()
    accuarcy_pca_knn = pca_knn()
    pca_mnist_perfor()
    accuarcy_gmm = gmm()
    accuarcy_svm = svm_t()
    spclus()
    accuarcy_spclu = best_spclu()