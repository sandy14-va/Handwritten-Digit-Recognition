
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np


# ## Loading the Dataset

(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()

x_train.shape

x_train=x_train.reshape(-1, 28, 28, 1)
x_test=x_test.reshape(-1, 28, 28, 1)

x_test.shape


# ## Normalizing the data

x_train=x_train/255
x_test=x_test/255


# ## Building a CNN 

cnn=keras.Sequential([
    #cnn layers
    keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)),
    keras.layers.MaxPool2D((2,2)),
    
    keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
    keras.layers.MaxPool2D((2,2)),
    
    keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
    keras.layers.MaxPool2D((2,2)),
    
    #Dense Layers
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

cnn.compile(optimizer="adam",
           loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
           )

cnn.fit(x_train, y_train, epochs=10, validation_data=(x_test,y_test))

y_pred= cnn.predict(x_test)
y_pred


cnn.evaluate(x_test, y_test)

y_pred_classes=[]
for el in y_pred:
    y_pred_classes.append(np.argmax(el))

y_test[:5]

y_pred_classes[:5]

cnn.save("mnist_conv_model.h5")