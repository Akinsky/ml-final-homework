import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def print_time(t1, t2, action_name):
    print(f'🕛 Timer | {action_name} : Executed in {(t2 - t1):.4f}s')
    print("--------------------------------------------------------------------")
    return t2 - t1


def prediction_heatmap(y_test, y_pred, y_names, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f",
                cmap="coolwarm", linewidths=1, linecolor="white",
                xticklabels=y_names, yticklabels=y_names)
    plt.savefig(f'./plots/predictions/{filename}.png')  # Save heatmap result to location as png.
    plt.close()
