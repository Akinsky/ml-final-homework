import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from my_data import MusicData
from time import time
from utility import print_time


def knn(k):
    data = MusicData()
    X_train, X_test, _, _ = data.get_x_y_split()
    y_train, y_test = data.get_y_categorical()
    y_train = y_train.values.ravel()  # Turns Y into an 1 Dimensional array.

    neighbors = np.arange(1, k)  # Iterate over k neighbours.
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    training_time = 0
    # Loop over K values
    t_start = time()  # Start timer for the whole execution.
    for i, k in enumerate(neighbors):
        t1 = time()  # Start timer for one iteration.
        print(f"Fitting KNN for n={k}...")

        # Training the KNN.
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Compute training and test data accuracy
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)

        # Print results
        print(f"Train Accuracy of KNN for k={k} is: {train_accuracy[i]}")
        print(f" Test Accuracy of KNN for k={k} is: {test_accuracy[i]}")
        t2 = time()  # Stop timer for one iteration.
        training_time = print_time(t1, t2, "Fitted KNN and calculated scores")


    # Generating the plot
    print(f"Generating the plot for the accuracies of the KNN...")
    plt.plot(neighbors, test_accuracy, label='Testing Dataset Accuracy', color='green')
    plt.plot(neighbors, train_accuracy, label='Training Dataset Accuracy', color='red')

    plt.legend()
    plt.xlabel('n_neighbours')
    plt.ylabel('Accuracy')
    plt.savefig('./plots/training/knn.png')
    plt.close()
    print(f"Plot generated successfully at location: ./plots/training/knn.png")

    t_end = time()  # Stop the whole timer.
    print_time(t_start, t_end, "Trained all KNN")

    best_index = test_accuracy.argmax()
    print(f"Best accuracy was for k={best_index + 1}, with the test accuracy of: {test_accuracy[best_index] * 100}")

    return test_accuracy[best_index] * 100, training_time

