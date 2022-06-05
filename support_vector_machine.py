from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from my_data import MusicData
from utility import print_time, prediction_heatmap
from time import time


def support_vector_machine(kernel):
    data = MusicData()
    X_train, X_test, _, _ = data.get_x_y_split()
    y_train, y_test = data.get_y_categorical()
    y_train = y_train.values.ravel()  # Turns Y into an 1 Dimensional array.

    t1 = time()
    print(f"Training the SVM model with '{kernel}' kernel...")
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)  # Train the model.
    t2 = time()

    training_time = print_time(t1, t2, "Trained the SVM Model")

    t1 = time()
    y_pred = clf.predict(X_test)  # Make predictions with the trained SVM model.

    accuracy = accuracy_score(y_test, y_pred) * 100

    print(f"The Accuracy for the Test Set is {accuracy}")
    print(classification_report(y_test, y_pred))
    prediction_heatmap(y_test, y_pred, data.get_y_names(), "support_vector_machine")
    t2 = time()

    print_time(t1, t2, "Predicted results")

    return accuracy, training_time
