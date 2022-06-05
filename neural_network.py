from my_data import MusicData
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from time import time
from utility import print_time

# TODO: Implement Timing
# TODO: Loss Graph

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5

def ann(loss, optimizer):
    data = MusicData()
    X_train, X_test, y_train, y_test = data.get_x_y_split()

    model = Sequential()
    model.add(Dense(256, input_dim=len(X_train.columns), activation='relu'))  # This defines 2 layers. The input layer and the first hidden layer.
    model.add(Dense(128, activation='relu'))  # 3rd Layer
    model.add(Dense(64, activation='relu'))  # 4th Layer
    model.add(Dense(32, activation='relu'))  # 5th Layer
    model.add(Dense(10, activation='softmax'))  # 6th Layer (Output)

    # Compile the model and calculate its accuracy: In this case, we will use cross entropy as the loss argument. This
    # loss is for a binary classification problems and is defined in Keras as “binary_crossentropy“.

    # We will define the optimizer as the efficient stochastic gradient descent algorithm “adam“. This is a popular
    # version of gradient descent because it automatically tunes itself and gives good results in a wide range of problems.
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model.summary()  # Print a summary of the Keras model:

    t1 = time()
    history = model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)
    t2 = time()
    training_time = print_time(t1, t2, "Trained the Artificial Neural Network Model")

    # Evaluate the keras model
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))

    return accuracy * 100, training_time

