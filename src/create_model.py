from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
import json

DATASET_PATH = "dataset.json"
MODEL_PATH = "model.h5"

NUM_COMMANDS = 8
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32

# load dataset from json file
with open(DATASET_PATH, "r") as dataset_data:
    dataset = json.load(dataset_data)

# extract inputs and targets
X = np.array(dataset["MFCCs"])
y = np.array(dataset["labels"])

# split dataset for train/validation/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, test_size=0.1)

# convert inputs from 2D to 3D arrays example:
# (number of segments, number of MFCCs, 1)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
X_validation = X_validation[..., np.newaxis]

# build network (CNN)
model = keras.Sequential()
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

# create first convolutional network
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu",
          input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

# create second convolutional network
model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
          kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

# create third convolutional network
model.add(keras.layers.Conv2D(32, (2, 2), activation="relu",
          kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

# flatten the output feed it into a dense layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dropout(0.3))

# softmax classifier
model.add(keras.layers.Dense(NUM_COMMANDS, activation="softmax"))

# compile the model
optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimiser,
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# train the model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_data=(X_validation, y_validation))

# evaluate the model
test_error, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test error: {test_error}, accuracy: {test_accuracy}")

# save the model
model.save(MODEL_PATH)
