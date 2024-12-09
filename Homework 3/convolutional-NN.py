# import libraries
import numpy as np
from utils import *
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

class ConvNN:
    def __init__(self, num_classes, num_samples_train, num_samples_test, seed, input_shape=(28, 28, 1), is_colab=False):
        self.num_classes = num_classes
        self.num_samples_train = num_samples_train
        self.num_samples_test = num_samples_test
        self.seed = seed
        self.input_shape = input_shape
        self.model = self.build_model()
        self.train_image, self.train_label, self.test_image, self.test_label = self.LoadData(num_classes, num_samples_train, num_samples_test, seed, is_colab)

    # The CNN Architecture, inspired by the one given in the homework
    def build_model(self):
        model = models.Sequential()

        # First Conv block: 28x28x1 -> 26x26x32 -> 26x26x32 -> 26x26x32 -> 13x13x32
        model.add(layers.Conv2D(32, (3, 3), padding='valid', input_shape=self.input_shape))
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
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        return model
    
    def LoadData(self, num_classes, num_samples_train, num_samples_test, seed, is_colab):
        train_image, train_label, test_image, test_label = LoadData(num_classes, num_samples_train, num_samples_test, seed, is_colab)

        return train_image, train_label, test_image, test_label
    
    def train_and_evaluate(self, model, learning_rate, epochs):
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()],
        )
        history = model.fit(
            self.train_image,
            self.train_label,
            epochs=epochs,
            validation_data=(self.test_image, self.test_label),
            verbose=0,
        )
        return history

    def train_model(self, epochs=500, alpha=1e-5):
        history = self.train_and_evaluate(self.model, learning_rate=alpha, epochs=epochs)
        return history
    
    def train_multiple_models(self, learning_rates, epochs, print_output=True, plot_graph=True):
        results = []
        for lr in learning_rates:
            for epoch in epochs:
                model = self.build_model()
                history = self.train_and_evaluate(model, learning_rate=lr, epochs=epoch)
                final_val_accuracy = history.history['val_sparse_categorical_accuracy'][-1]

                if print_output:
                    print(f"{final_val_accuracy:.3f}: {lr:.3f} x {epoch}")
                results.append((lr, epoch, final_val_accuracy))

        if plot_graph:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            learning_rates_plot, epoch_values_plot, val_accuracies_plot = zip(*results)

            sc = ax.scatter(learning_rates_plot, epoch_values_plot, val_accuracies_plot, c=val_accuracies_plot, cmap='viridis', s=100)
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Epochs')
            ax.set_zlabel('Validation Accuracy')
            plt.colorbar(sc, label='Validation Accuracy')
            plt.title('Learning Rate vs Epochs vs Validation Accuracy')
            plt.show()
        return results

    def show_n_predictions(self, n=10, print_output=True):
        for i in range(n):
            index = np.random.randint(0, self.test_image.shape[0])
            image = self.test_image[index].reshape(1, 28, 28, 1)
            prediction = self.model.predict(image)
            print(f"Predicted class: {np.argmax(prediction)}, Actual class: {self.test_label[index]}")

        if(print_output):
            test_loss, test_acc = self.model.evaluate(self.test_image, self.test_label, verbose=2)
            print(f"Test accuracy: {test_acc}")
            print(f"Test loss: {test_loss}")

# Build and summarize the model
model = ConvNN(num_classes=50, num_samples_train=15, num_samples_test=5, seed=1)

# Define ranges for learning rate and epochs
learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
epoch_values = [10, 50, 100, 500, 1000]


# Plotting the 3D graph
model.train_multiple_models(learning_rates, epoch_values, print_output=True, plot_graph=True)