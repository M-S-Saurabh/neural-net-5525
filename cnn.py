import numpy as np
import time

import os
# Disable printing all the tensorflow information at the start
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import plot_loss_accuracy, plot_runtime_vs_minibatch

IMG_WIDTH = 28

def get_processed_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float') / 255
    x_test = x_test.astype('float') / 255
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = get_processed_data()

def build_CNN(modelname):
    inputs = keras.Input(shape=(IMG_WIDTH, IMG_WIDTH, 1))
    # Convolutional layer 1
    cv1 = layers.Conv2D(20, 3, strides=(1,1), padding='same', activation='relu')(inputs)
    # Max pooling
    mp1 = layers.MaxPool2D(pool_size=(2,2), padding='same')(cv1)
    # Dropout 1
    drop1 = layers.Dropout(0.5, seed=42)(mp1)
    # Flatten
    flat1 = layers.Flatten()(drop1)
    # Fully connected 1
    fc1 = layers.Dense(128, activation='relu')(flat1)
    # Dropout 2
    drop2 = layers.Dropout(0.5, seed=42)(fc1)
    # Fully connected 2
    outputs = layers.Dense(10, activation='softmax')(drop2)
    # Model
    model = keras.Model(inputs=inputs, outputs=outputs, name=modelname)
    return model

def runtime_vs_minibatch(model, optimizer_name, epochs):
    if optimizer_name == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=0.01)
    elif optimizer_name == 'Adagrad':
        optimizer = keras.optimizers.Adagrad(learning_rate=0.01)
    else:
        #elif optimizer_name == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=['accuracy']
    )
    batchsizes = [32, 64, 96, 128]
    runtimes = []
    monitor = 'loss'; min_delta = 1e-4
    earlystopping = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta)
    print("=============================================")
    print("Running the model with {} optimizer".format(optimizer_name))
    print("Stopping criteria: {} < {}".format(monitor, min_delta))
    for batchsize in batchsizes:
        print("Training with batch size={} until convergence...".format(batchsize), end='\r', flush=True)
        start_time = time.time()
        history = model.fit(x_train, y_train, batch_size=batchsize, epochs=epochs, callbacks=[earlystopping], verbose=0)
        runtime = time.time() - start_time
        runtimes.append(runtime)
        print("Training with batch size={} took {:.2f} secs      ".format(batchsize, runtime))
        print("num epochs:", len(history.history['loss']))
    plot_runtime_vs_minibatch(runtimes, batchsizes, optimizer_name)

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

def cnn() -> None:
    np.random.seed(42)
    modelname = 'CNN'
    model = build_CNN(modelname)
    # Train a model using SGD and measure traning loss, testing accuracy
    # loss_accuracy_vs_epochs(model, modelname)
    # Measure convergence time vs batchsize
    epochs=30
    runtime_vs_minibatch(model, 'SGD', epochs)
    runtime_vs_minibatch(model, 'Adagrad', epochs)
    runtime_vs_minibatch(model, 'Adam', epochs)

if __name__ == "__main__":
    cnn()