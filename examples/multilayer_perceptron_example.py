import numpy as np

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlib.algorithms.multilayer_perceptron import MultilayerPerceptron
from mlib.utils import vector_to_one_hot_matrix

def multilayer_perceptron_example():
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    num_classes = len(np.unique(y_train))

    mlp_nn = MultilayerPerceptron(
        num_input_nodes=X_train.shape[1],
        num_output_nodes=num_classes,
        num_hidden_nodes=150,
        learning_rate = 0.001,
        mini_batch_size = 32,
        type="class"
    )

    mlp_nn.fit(X_train, vector_to_one_hot_matrix(y_train, num_classes), 25)
    y_pred = mlp_nn.predict(X_test)
    print('pred:', y_pred)

    acc_score = accuracy_score(y_test, np.argmax(y_pred, axis=1))
    print('accuracy: ', acc_score)

multilayer_perceptron_example()