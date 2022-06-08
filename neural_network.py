from time import time

from ann_visualizer.visualize import ann_viz;
from keras.layers import Dense
from keras.models import Sequential

from my_data import MusicData
from utility import print_time


def ann(loss, optimizer, epochs, batch_size):
    data = MusicData()
    X_train, X_test, y_train, y_test = data.get_x_y_split()

    model = Sequential()
    model.add(Dense(256, input_dim=len(X_train.columns), activation='relu'))  # This defines 2 layers. The input layer and the first hidden layer.
    model.add(Dense(128, activation='relu'))  # 3rd Layer
    model.add(Dense(64, activation='relu'))  # 4th Layer
    model.add(Dense(32, activation='relu'))  # 5th Layer
    model.add(Dense(10, activation='softmax'))  # 6th Layer (Output)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model.summary()  # Print a summary of the Keras model:

    t1 = time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    t2 = time()
    training_time = print_time(t1, t2, "Trained the Artificial Neural Network Model")

    # Evaluate the Keras Model
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))

    try:
        ann_viz(model, title="Neural Network Drawing")
    except:
        print("Could not draw the Neural Network visualization: Permission Denied.")

    return accuracy * 100, training_time