import matplotlib.pyplot as plt
import pandas as pd

from decision_tree import dtree
from knn import knn
from logistic_regression import logistic_regression
from neural_network import ann
from support_vector_machine import support_vector_machine

model_order = ['SVM', 'LR', 'KNN', 'DTree', 'ANN']
accuracies = []
times = []

accuracy, time = support_vector_machine('rbf')
accuracies.append(accuracy)
times.append(time)

accuracy, time = logistic_regression(1000)
accuracies.append(accuracy)
times.append(time)

accuracy, time = knn(20)
accuracies.append(accuracy)
times.append(time)

accuracy, time = dtree()
accuracies.append(accuracy)
times.append(time)

accuracy, time = ann('binary_crossentropy', 'adam', 10, 8)
accuracies.append(accuracy)
times.append(time)

df = pd.DataFrame({'accuracies': accuracies, 'times': times}, index=model_order)
df.sort_values(inplace=True, by='accuracies')
ax = df.plot.barh(cmap='Set1')
ax.set_title("Models Sorted by Best Accuracy")
ax.set_xlabel("Performance")
plt.savefig("./plots/predictions/model_comparison_1.png")
plt.close()
print(df.head())
