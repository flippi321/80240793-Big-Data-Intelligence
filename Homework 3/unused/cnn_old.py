# import libraries
import argparse
import numpy as np
import torch
from utils import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

# The CNN Architecture, inspired by the one given in the homework
def build_model(num_classes):
    model = models.Sequential()

    # First Conv block: 28x28x1 -> 26x26x32 -> 26x26x32 -> 26x26x32 -> 13x13x32
    model.add(layers.Conv2D(32, (3, 3), padding='valid', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(32, (3, 3), padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Second Conv block: 13x13x32 -> 11x11x64 -> 11x11x64 -> 11x11x64 -> 5x5x64
    model.add(layers.Conv2D(64, (3, 3), padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(64, (3, 3), padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Fully connected layers
    # Flatten: 5x5x64 -> 1600
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def getData():
    print("Test")
    

# define settings
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=50, help='number of classes used')
parser.add_argument('--num_samples_train', type=int, default=15, help='number of samples per class used for training')
parser.add_argument('--num_samples_test', type=int, default=5, help='number of samples per class used for testing')
parser.add_argument('--seed', type=int, default=1, help='random seed')
args = parser.parse_args()

# Settings
epochs = 500
alpha = 25e-6

# load data
train_image, train_label, test_image, test_label = LoadData(args.num_classes, args.num_samples_train, args.num_samples_test, args.seed)
print(train_label[0])


# I should probably fix the original functions
train_image = train_image.reshape(-1, 28, 28, 1)
test_image = test_image.reshape(-1, 28, 28, 1) 

array_to_image_file(train_image[0], "output_image.png", [28, 28])
print(train_image[0])
print(train_label[0])

# Build and summarize the model
model = build_model(num_classes=args.num_classes)
# model.summary()

# Train the model
model.compile(
    optimizer=Adam(learning_rate=alpha), 
    loss=SparseCategoricalCrossentropy(), 
    metrics=[
        SparseCategoricalAccuracy()
        ]
    )

history = model.fit(train_image, train_label, epochs=epochs, validation_data=(test_image, test_label))

# Show the predicted and actual values for 5 random test images
for i in range(5):
    index = np.random.randint(0, test_image.shape[0])
    image = test_image[index].reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    print(f"Predicted class: {np.argmax(prediction)}, Actual class: {test_label[index]}")


test_loss, test_acc = model.evaluate(test_image, test_label, verbose=2)