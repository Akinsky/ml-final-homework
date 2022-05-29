from sklearn import metrics
from sklearn import svm

from my_data import MusicData

data = MusicData()
df = data.get_dataframe()
X_train, X_test, _, _ = data.get_x_y_split()
y_train, y_test = data.get_y_categorical()
y_train = y_train.values.ravel()

print("Training the model...")
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)
