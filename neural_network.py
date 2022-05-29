from my_data import MusicData
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5

data = MusicData()
df = data.get_dataframe()
X_train, X_test, y_train, y_test = data.get_x_y_split()
# y_train, y_test = data.get_y_as_single_column()

# define the keras model
model = Sequential()
model.add(Dense(16, input_dim=len(X_train.columns), activation='relu'))  # This defines 2 layers. The input layer and the first hidden layer.
model.add(Dense(16, activation='relu'))  # 3rd Layer
model.add(Dense(10, activation='softmax'))  # 4th Layer (Output)

# Compile the model and calculate its accuracy: In this case, we will use cross entropy as the loss argument. This
# loss is for a binary classification problems and is defined in Keras as “binary_crossentropy“.

# We will define the optimizer as the efficient stochastic gradient descent algorithm “adam“. This is a popular
# version of gradient descent because it automatically tunes itself and gives good results in a wide range of problems.
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print a summary of the Keras model:
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))

