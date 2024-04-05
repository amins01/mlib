from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlib.algorithms.knn import KNN

def knn_example():
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    knn = KNN(k=2)
    knn.fit(X_train, y_train)
    y_pred = knn.predict_class(X_test)
    print('pred:', y_pred)

    acc_score = accuracy_score(y_test, y_pred)
    print('accuracy: ', acc_score)

knn_example()