import numpy as np

from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from mlib.algorithms.multilayer_perceptron import MultilayerPerceptron

def multilayer_perceptron_example():
    housing = datasets.fetch_california_housing()

    X = housing.data
    y = housing.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    mlp_nn = MultilayerPerceptron(
        num_input_nodes=X_train.shape[1],
        num_output_nodes=1,
        num_hidden_nodes=250,
        learning_rate = 0.0001,
        mini_batch_size = 32,
        type="reg"
    )

    mlp_nn.fit(X_train, np.reshape(y_train, (len(y_train), 1)), 50)
    y_pred = mlp_nn.predict(X_test)
    print('test:', y_test)
    print('pred:', np.reshape(y_pred, (y_pred.shape[0],)))

    r2 = r2_score(y_test, np.reshape(y_pred, (y_pred.shape[0],)))
    print('R^2 Score: ', r2)

multilayer_perceptron_example()