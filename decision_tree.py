import pydotplus
from IPython.display import Image
from six import StringIO
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.tree import export_graphviz
from my_data import MusicData
from time import time
from utility import print_time


def dtree():
    global training_time
    global accuracy

    def visualize_tree(decision_tree, filename):
        """ Draws the tree graph"""
        dot_data = StringIO()
        export_graphviz(decision_tree, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=X_train.columns)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(f'./plots/training/{filename}.png')
        Image(graph.create_png())

    data = MusicData(dtree=True)
    X_train, X_test, y_train, y_test = data.get_x_y_split()

    print(X_train, X_test, y_train, y_test)

    best_depth = 3
    old_accuracy = 0
    best_tree = None

    for depth in range(1, 13):
        t1 = time()
        dtree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        dtree = dtree.fit(X_train, y_train)
        t2 = time()
        training_time = print_time(t1, t2, "Trained Decision Tree")

        y_pred = dtree.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred) * 100
        print(f"DEPTH = {depth} | Accuracy: {accuracy}%")

        if accuracy > old_accuracy:
            print(f"Found better accuracy at this depth! Old Acc: {old_accuracy} new Acc: {accuracy}")
            old_accuracy = accuracy
            best_depth = depth
            best_tree = dtree

    print(f"Best accuracy was found at depth {best_depth}. Generating graph...")
    visualize_tree(best_tree, f"dtree_best")

    dtree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    visualize_tree(dtree, "dtree_depth=3")

    return accuracy, training_time

    """
    Gini ratio, which measures the impurity of the node. 
    
    Here, the resultant tree is unpruned. This unpruned tree is unexplainable and not easy to understand. In the next section, let's optimize it by pruning.
    
    value = [6, 7] means that of these 13 comedians, 6 will get a "NO", and 7 will get a "GO".
    
    samples = 13 means that there are 13 comedians left at this point in the decision, which is all of them since this is the first step.
    
    Rank <= 6.5 means that every comedian with a rank of 6.5 or lower will follow the True arrow (to the left), and the rest will follow the False arrow (to the right).
    """
    # https://www.w3schools.com/python/python_ml_decision_tree.asp
    # https://www.datacamp.com/tutorial/decision-tree-classification-python
