import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

# %matplotlib inline

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
# train.head()

X_train = (train.iloc[:, 1:].values).astype("float32")
y_train = (train.iloc[:, 0].values).astype("int32")
X_test = test.values.astype("float32")

X_train = X_train.reshape(X_train.shape[0], 28, 28)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x):
    return (x - mean_px) / std_px

y_train = to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes

X_train, X_val, y_train, y_val = train_test_split(X_train[:100], y_train[:100], test_size=0.2)
# gen = ImageDataGenerator()
# train_batches = gen.flow(X_train, y_train, batch_size=64)
# val_batches = gen.flow(X_val, y_val, batch_size=64)

# def get_cnn_model():
#     model = Sequential([
#         Lambda(standardize, input_shape=(28, 28, 1)),
#         Convolution2D(32, (3,3), activation="relu"),
#         Convolution2D(32, (3,3), activation="relu"),
#         MaxPooling2D(),
#         Convolution2D(64, (3,3), activation="relu"),
#         Convolution2D(64, (3,3), activation="relu"),
#         MaxPooling2D(),
#         Flatten(),
#         Dense(512, activation="relu"),
#         Dense(10, activation="softmax"),
#     ])
#     model.compile(Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
#     return model

# model = get_cnn_model()
# model.optimizer.lr = 0.01

# hitory = model.fit_generator(
#     generator=train_batches, 
#     steps_per_epoch=train_batches.n,
#     epochs=1,
#     validation_data=val_batches,
#     validation_steps=val_batches.n,
# )

gen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    shear_range=0.3,
    height_shift_range=0.08,
    zoom_range=0.08,
)
train_batches = gen.flow(X_train, y_train, batch_size=64)
val_batches = gen.flow(X_val, y_val, batch_size=64)

def get_cnn_model_with_bn():
    model = Sequential([
        Lambda(standardize, input_shape=(28, 28, 1)),
        Convolution2D(32, (3,3), activation="relu"),
        BatchNormalization(axis=1),
        Convolution2D(32, (3,3), activation="relu"),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64, (3,3), activation="relu"),
        BatchNormalization(axis=1),
        Convolution2D(64, (3,3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dense(10, activation="softmax"),
    ])
    model.compile(Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = get_cnn_model_with_bn()
model.optimizer.lr = 0.01

hitory = model.fit_generator(
    generator=train_batches, 
    steps_per_epoch=train_batches.n,
    epochs=3,
    validation_data=val_batches,
    validation_steps=val_batches.n,
)

predictions = model.predict_classes(X_test, verbose=0)
submissions = pd.DataFrame({
    "ImageId": list(range(1, len(predictions) + 1)),
    "Label": predictions,
})
submissions.to_csv("./output/submission_05.csv", index=False)

