from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import data_reader
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def check_k(n, X, Y, Xtest, Ytest):
    k = []
    scores = []
    errors = []
    for K in range(n):
        k_value = K + 1
        k_neighbor = KNeighborsClassifier(n_neighbors=k_value, weights='uniform', algorithm='auto')
        k_neighbor.fit(X, Y)
        y_pred = k_neighbor.predict(Xtest)
        k.append(k_value)
        scores.append(metrics.accuracy_score(Ytest, y_pred) * 100)
        errors.append(1 - metrics.accuracy_score(Ytest, y_pred))

    plt.plot(k, scores)
    plt.xlabel('k-value')
    plt.ylabel('Accuracy')
    plt.title('K - Accuracy')
    plt.grid()
    plt.show()

    plt.plot(k, errors)
    plt.xlabel('k-value')
    plt.ylabel('Errors')
    plt.title('K - Errors')
    plt.grid()
    plt.show()


def get_solution():
    X_train, X_test = data_reader.read_data()

    x_train, x_test = X_train.drop('wage_class', axis=1), X_test.drop('wage_class', axis=1)
    y_train, y_test = X_train['wage_class'], X_test['wage_class']

    X, Y = x_train.values, y_train.values

    Xtest, Ytest = x_test.values, y_test.values

    #check_k(35, X, Y, Xtest, Ytest)

    kNN = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto')
    kNN.fit(X, Y)
    Ypred = kNN.predict(Xtest)

    ConfusionMatrixDisplay.from_predictions(Ytest, Ypred, cmap='binary')
    plt.title('Confusion matrix')
    plt.show()
    print('Accuracy = {}\n'
          'Error = {}'.format(metrics.accuracy_score(Ytest, Ypred), 1 - metrics.accuracy_score(Ytest, Ypred)))
