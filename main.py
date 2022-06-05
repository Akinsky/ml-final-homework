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

accuracy, time = ann('binary_crossentropy', 'adam')
accuracies.append(accuracy)
times.append(time)

""" REAL RUN 1 RobustScaler Populated 
       accuracies      times
KNN     49.815089   4.329000
DTree   50.936884   2.771217
LR      55.399408  14.931998
SVM     57.593688  60.287024
ANN     59.134614  70.788965

REAL RUN 2 RobustScaler Non Populated
       accuracies      times
KNN     48.915187   3.728407
DTree   51.084813   1.125275
LR      53.846154   3.957327
SVM     57.988166  61.200604
ANN     58.222389  73.825566

REAL RUN 3 MinMaxScaler Non Populated
       accuracies      times
KNN     44.995069   4.013001
DTree   51.023176   1.204003
LR      53.759862   8.193000
SVM     54.857002  63.022998
ANN     56.669134  73.772000

REAL RUN 4 StandartScaler Non Populated
       accuracies      times
ANN     19.045858  82.564559
KNN     48.089250   4.357453
DTree   50.936884   2.031972
LR      53.870809   1.592998
SVM     56.508876  79.743391
"""

df = pd.DataFrame({'accuracies': accuracies, 'times': times}, index=model_order)
df.sort_values(inplace=True, by='accuracies')
ax = df.plot.barh(cmap='Set1')
ax.set_title("Models Sorted by Best Accuracy")
ax.set_xlabel("Performance")
plt.savefig("./plots/predictions/model_comparison_robust_nopopulate.png")
plt.close()

print(df.head())
