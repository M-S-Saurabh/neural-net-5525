import numpy as np

import os
# Disable printing all the tensorflow information at the start
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import plot_loss_accuracy

INPUT_SIZE = 28*28

def get_processed_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, INPUT_SIZE).astype('float') / 255
    x_test = x_test.reshape(-1, INPUT_SIZE).astype('float') / 255
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = get_processed_data()

def loss_accuracy_vs_epochs(model, modelname, epochs=10):
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        metrics=['accuracy']
    )
    history = model.fit(x_train, y_train, batch_size=32, epochs=epochs)
    history = history.history
    # Training loss and accuracy vs epochs
    plot_loss_accuracy(history['loss'], history['accuracy'], modelname)
    print("----------------------------------")
    print("Training loss")
    print(np.round(history['loss'], 3))
    print("----------------------------------")
    print("Training accuracy")
    print(np.round(history['accuracy'], 3))
    # Testing accuracy reported as single value
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('------------------------------------------------')
    print("Test accuracy:", test_scores[1])

def neural_net() -> None:
    inputs = keras.Input(shape=(INPUT_SIZE,))
    # Fully connected layer 1
    fc1 = layers.Dense(128, activation='relu')
    x = fc1(inputs)
    # Fully connected layer 2
    fc2 = layers.Dense(10, activation='softmax')
    outputs = fc2(x)
    # Model
    modelname = 'FC-NN'
    model = keras.Model(inputs=inputs, outputs=outputs, name=modelname)

    # Train a model using SGD and measure traning loss, testing accuracy
    loss_accuracy_vs_epochs(model, modelname)

if __name__ == "__main__":
    neural_net()