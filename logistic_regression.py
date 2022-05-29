from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from my_data import MusicData

# https://asperbrothers.com/blog/logistic-regression-in-python/

data = MusicData()
df = data.get_dataframe()
X_train, X_test, _, _ = data.get_x_y_split()
y_train, y_test = data.get_y_categorical()

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("The Accuracy for Test Set is {}".format(test_acc * 100))

print(classification_report(y_test, y_pred))
