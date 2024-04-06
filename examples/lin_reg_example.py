from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from mlib.algorithms.linear_regression import LinearRegression

def lin_reg_example():
    housing = datasets.fetch_california_housing()

    X = housing.data
    y = housing.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    lin_reg = LinearRegression()

    lin_reg.fit(X_train, y_train)
    
    y_pred = lin_reg.predict_reg(X_test)
    print('truth:', y_test)
    print('pred:', y_pred)

    r2 = r2_score(y_test, y_pred)
    print('R^2 Scrore: ', r2)

lin_reg_example()