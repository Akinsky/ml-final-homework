import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from math import sqrt

from my_data import MusicData

data = MusicData()
df = data.get_dataframe()
X_train, X_test, y_train, y_test = data.get_x_y_split()
y_train, y_test = data.get_y_as_single_column()
y_train = y_train.values.ravel()
# knn_model = KNeighborsRegressor(n_neighbors=15)
# knn_model.fit(X_train, y_train)
# y_pred = knn_model.predict(X_test)
# Calculate the accuracy of the model
# print(knn_model.score(X_test, y_test)*100)

neighbors = np.arange(1, 25)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.savefig('./plots/knn.png')
plt.show()

