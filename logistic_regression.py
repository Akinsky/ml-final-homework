from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from my_data import MusicData
from utility import print_time, prediction_heatmap
from time import time


def logistic_regression(iterations):
    data = MusicData()
    X_train, X_test, _, _ = data.get_x_y_split()
    y_train, y_test = data.get_y_categorical()
    y_train = y_train.values.ravel()

    t1 = time()
    model = LogisticRegression(max_iter=iterations)
    model.fit(X_train, y_train)
    t2 = time()
    training_time = print_time(t1, t2, "Trained the Logistic Regression Model")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"The Accuracy for the Test Set is {accuracy}")
    print(classification_report(y_test, y_pred))
    prediction_heatmap(y_test, y_pred, data.get_y_names(), "logistic_regression")

    return accuracy, training_time
