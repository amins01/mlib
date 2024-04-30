from sklearn import datasets
from sklearn.model_selection import train_test_split

from mlib.algorithms.k_means import KMeans

def k_means_example():
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    k_means = KMeans(k=3)
    k_means.fit(X_train)
    pred = k_means.predict(X_test)
    print('pred:', pred)

k_means_example()