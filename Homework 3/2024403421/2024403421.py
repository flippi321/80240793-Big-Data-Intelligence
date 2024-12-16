import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
from tensorflow import keras
from keras import layers, models
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy


# ----- Utils ----- 
def LoadData(num_classes = 50, num_samples_per_class_train = 15, num_samples_per_class_test = 5, seed = 1, flatten=False, is_colab=False):
    """
    Load data and split it into training and testing
    Args:
        num_classes: number of classes adopted, -1 represents using all the classes
        num_samples_per_class_train: number of samples per class used for training
        num_samples_per_class_test: number of samples per class used for testing
        seed: random seed to ensure consistent results
    Returns:
        a tuple of (1) images for training (2) labels for training (3) images for testing, and (4) labels for testing
            (1) numpy array of shape [num_classes * num_samples_per_class_train, 784], binary pixels
            (2) numpy array of shape [num_classes * num_samples_per_class_train], integers of the class label
            (3) numpy array of shape [num_classes * num_samples_per_class_test, 784], binary pixels
            (4) numpy array of shape [num_classes * num_samples_per_class_test], integers of the class label
    """
    random.seed(seed)
    np.random.seed(seed)
    num_samples_per_class = num_samples_per_class_train + num_samples_per_class_test
    assert num_classes <= 1623
    assert num_samples_per_class <= 20

    # construct folders
    data_folder = '/content/drive/MyDrive/Colab Notebooks/Big Data Intelligence/data/omniglot_resized' if is_colab else './omniglot_resized'
    character_folders = [os.path.join(data_folder, family, character)
                         for family in os.listdir(data_folder)
                         if os.path.isdir(os.path.join(data_folder, family))
                         for character in os.listdir(os.path.join(data_folder, family))
                         if os.path.isdir(os.path.join(data_folder, family, character))]
    random.shuffle(character_folders)
    if num_classes == -1:
        num_classes = len(character_folders)
    else:
        character_folders = character_folders[:num_classes]

    # read images
    all_images = np.zeros(shape=(num_samples_per_class, num_classes, 28, 28))
    all_labels = np.zeros(shape=(num_samples_per_class, num_classes))
    label_images = get_images(character_folders, list(range(num_classes)), nb_samples=num_samples_per_class, shuffle=True)
    temp_count = np.zeros(num_classes, dtype=int)
    for label, imagefile in label_images:
        temp_num = temp_count[label]
        all_images[temp_num, label, :, :] = image_file_to_array(imagefile)
        all_labels[temp_num, label] = label
        temp_count[label] += 1

    # split and random permutate
    train_image = all_images[:num_samples_per_class_train].reshape(-1, 28, 28)
    test_image = all_images[num_samples_per_class_train:].reshape(-1, 28, 28)
    train_label = all_labels[:num_samples_per_class_train].reshape(-1)
    test_label = all_labels[num_samples_per_class_train:].reshape(-1)
    train_image, train_label = pair_shuffle(train_image, train_label)
    test_image, test_label = pair_shuffle(test_image, test_label)

    if(flatten):
        train_image = train_image.reshape(-1, 784)
        test_image = test_image.reshape(-1, 784)

    return train_image, train_label, test_image, test_label


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler([pathstr for pathstr in os.listdir(path) if pathstr[-4:] == '.png' ])]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
    Returns:
        1 channel image
    """
    # Updated code as function was depricated
    image = Image.open(filename).convert('L')
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image

def array_to_image_file(image_array, filename, dim_output):
    """
    Takes a numpy array and saves it as an image file.
    Args:
        image_array: Flattened numpy array of the image.
        filename: File path to save the image (including extension, e.g., .png).
        dim_output: Shape of the output image (e.g., [height, width] for 2D images).
    """
    # Reshape the array to the desired dimensions (height x width)
    image = image_array.reshape(dim_output)

    # Reverse the transformation from image_file_to_array
    image = 1.0 - image
    image = (image * 255).astype(np.uint8)

    # We can now use PIL to save the image
    pil_image = Image.fromarray(image)
    pil_image.save(filename)

def pair_shuffle(array_a, array_b):
    """
    Takes an image array and a label array
    Returns:
        the shuffled image array and label array
    """
    temp_perm = np.random.permutation(array_a.shape[0])
    array_a = array_a[temp_perm]
    array_b = array_b[temp_perm]
    return array_a, array_b

# ----- Convolutional Neural Network ----- 
class ConvNN:
    def __init__(self, num_classes, num_samples_train, num_samples_test, seed, input_shape=(28, 28, 1), is_colab=False):
        self.num_classes = num_classes
        self.num_samples_train = num_samples_train
        self.num_samples_test = num_samples_test
        self.seed = seed
        self.input_shape = input_shape
        self.model = self.build_model()
        self.train_image, self.train_label, self.test_image, self.test_label = LoadData(num_classes, num_samples_train, num_samples_test, seed, False, is_colab)

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

    def train_multiple_models(self, learning_rates, epochs, print_output=True, predictions=0, plot_graph=True):
        results = []
        for lr in learning_rates:
            for epoch in epochs:
                model = self.build_model()
                history = self.train_and_evaluate(model, learning_rate=lr, epochs=epoch)
                final_val_accuracy = history.history['val_sparse_categorical_accuracy'][-1]

                if print_output:
                    print(f"{final_val_accuracy:.3f}: {lr:.3f} x {epoch}")
                results.append((lr, epoch, final_val_accuracy))

                if predictions > 0:
                    for i in range(predictions):
                        index = np.random.randint(0, self.test_image.shape[0])
                        image = self.test_image[index].reshape(1, 28, 28, 1)
                        predicted_class = np.argmax(model.predict(image, verbose=0), axis=1)
                        print(f"Prediction: {predicted_class[0]}, Actual: {self.test_label[index]}")

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
            predicted_class = np.argmax(self.model.predict(image, verbose=0), axis=1)
            print(f"Predicted class: {predicted_class}, Actual class: {self.test_label[index]}")

        if(print_output):
            test_loss, test_acc = self.model.evaluate(self.test_image, self.test_label, verbose=2)
            print(f"Test accuracy: {test_acc}")
            print(f"Test loss: {test_loss}")

# ----- Fully Connected Neural Network ----- 
class ConnectedNN_1_layers:
    def __init__(self, num_classes, num_samples_train, num_samples_test, seed, is_colab=False):
        self.num_classes = num_classes
        self.num_samples_train = num_samples_train
        self.num_samples_test = num_samples_test
        self.seed = seed
        self.model = self.build_model()
        self.train_image, self.train_label, self.test_image, self.test_label = LoadData(num_classes, num_samples_train, num_samples_test, seed, True, is_colab)

    # The FCNN Architecture
    def build_model(self):
        model = models.Sequential()

        # Update input shape to match flattened input
        model.add(layers.Input(shape=(784,)))  # Flattened shape of 28x28

        # Add Dense layers with ReLU activation
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.3))

        # Output layer with softmax for classification
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        return model

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

    def train_multiple_models(self, learning_rates, epochs, print_output=True, predictions=0, plot_graph=True):
        results = []
        for lr in learning_rates:
            for epoch in epochs:
                model = self.build_model()
                history = self.train_and_evaluate(model, learning_rate=lr, epochs=epoch)
                final_val_accuracy = history.history['val_sparse_categorical_accuracy'][-1]

                if print_output:
                    print(f"{final_val_accuracy:.3f}: {lr:.3f} x {epoch}")
                results.append((lr, epoch, final_val_accuracy))

                if predictions > 0:
                  for i in range(predictions):
                      index = np.random.randint(0, self.test_image.shape[0])
                      image = self.test_image[index].reshape(1, 784)
                      predicted_class = np.argmax(model.predict(image, verbose=0), axis=1)
                      print(f"Prediction: {predicted_class[0]}, Actual: {self.test_label[index]}")



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
            image = self.test_image[index].reshape(1, 784)
            prediction = self.model.predict(image, verbose=0)
            print(f"Predicted class: {np.argmax(prediction)}, Actual class: {self.test_label[index]}")

        if(print_output):
            test_loss, test_acc = self.model.evaluate(self.test_image, self.test_label, verbose=2)
            print(f"Test accuracy: {test_acc}")
            print(f"Test loss: {test_loss}")

class ConnectedNN_2_layers:
    def __init__(self, num_classes, num_samples_train, num_samples_test, seed, is_colab=False):
        self.num_classes = num_classes
        self.num_samples_train = num_samples_train
        self.num_samples_test = num_samples_test
        self.seed = seed
        self.model = self.build_model()
        self.train_image, self.train_label, self.test_image, self.test_label = LoadData(num_classes, num_samples_train, num_samples_test, seed, True, is_colab)

    # The FCNN Architecture
    def build_model(self):
        model = models.Sequential()

        # Update input shape to match flattened input
        model.add(layers.Input(shape=(784,)))  # Flattened shape of 28x28

        # Add Dense layers with ReLU activation
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.3))

        # Output layer with softmax for classification
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        return model

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

    def train_multiple_models(self, learning_rates, epochs, print_output=True, predictions=0, plot_graph=True):
        results = []
        for lr in learning_rates:
            for epoch in epochs:
                model = self.build_model()
                history = self.train_and_evaluate(model, learning_rate=lr, epochs=epoch)
                final_val_accuracy = history.history['val_sparse_categorical_accuracy'][-1]

                if print_output:
                    print(f"{final_val_accuracy:.3f}: {lr:.3f} x {epoch}")
                results.append((lr, epoch, final_val_accuracy))

                if predictions > 0:
                  for i in range(predictions):
                      index = np.random.randint(0, self.test_image.shape[0])
                      image = self.test_image[index].reshape(1, 784)
                      predicted_class = np.argmax(model.predict(image, verbose=0), axis=1)
                      print(f"Prediction: {predicted_class[0]}, Actual: {self.test_label[index]}")



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
            image = self.test_image[index].reshape(1, 784)
            prediction = self.model.predict(image, verbose=0)
            print(f"Predicted class: {np.argmax(prediction)}, Actual class: {self.test_label[index]}")

        if(print_output):
            test_loss, test_acc = self.model.evaluate(self.test_image, self.test_label, verbose=2)
            print(f"Test accuracy: {test_acc}")
            print(f"Test loss: {test_loss}")

class ConnectedNN_3_layers:
    def __init__(self, num_classes, num_samples_train, num_samples_test, seed, is_colab=False):
        self.num_classes = num_classes
        self.num_samples_train = num_samples_train
        self.num_samples_test = num_samples_test
        self.seed = seed
        self.model = self.build_model()
        self.train_image, self.train_label, self.test_image, self.test_label = LoadData(num_classes, num_samples_train, num_samples_test, seed, True, is_colab)

    # The FCNN Architecture
    def build_model(self):
        model = models.Sequential()

        # Update input shape to match flattened input
        model.add(layers.Input(shape=(784,)))  # Flattened shape of 28x28

        # Add Dense layers with ReLU activation
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(128, activation='relu'))
        #model.add(layers.Dropout(0.3))
        #model.add(layers.Dense(64, activation='relu'))

        # Output layer with softmax for classification
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        return model

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

    def train_multiple_models(self, learning_rates, epochs, print_output=True, predictions=0, plot_graph=True):
        results = []
        for lr in learning_rates:
            for epoch in epochs:
                model = self.build_model()
                history = self.train_and_evaluate(model, learning_rate=lr, epochs=epoch)
                final_val_accuracy = history.history['val_sparse_categorical_accuracy'][-1]

                if print_output:
                    print(f"{final_val_accuracy:.3f}: {lr:.3f} x {epoch}")
                results.append((lr, epoch, final_val_accuracy))

                if predictions > 0:
                  for i in range(predictions):
                      index = np.random.randint(0, self.test_image.shape[0])
                      image = self.test_image[index].reshape(1, 784)
                      predicted_class = np.argmax(model.predict(image, verbose=0), axis=1)
                      print(f"Prediction: {predicted_class[0]}, Actual: {self.test_label[index]}")



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
            image = self.test_image[index].reshape(1, 784)
            prediction = self.model.predict(image, verbose=0)
            print(f"Predicted class: {np.argmax(prediction)}, Actual class: {self.test_label[index]}")

        if(print_output):
            test_loss, test_acc = self.model.evaluate(self.test_image, self.test_label, verbose=2)
            print(f"Test accuracy: {test_acc}")
            print(f"Test loss: {test_loss}")


# ----- Main ----- 

# Build and summarize the model
model_conv = ConvNN(num_classes=50, num_samples_train=15, num_samples_test=5, seed=1, is_colab=True)

# Define ranges for ConvNN
# 0.024: 0.500 x 750
# 0.592: 0.100 x 250
# 0.856: 0.050 x 750
learning_rates_conv = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
epoch_values_conv = [500, 1000, 2000]

"""
[(0.05, 500, 0.8960000276565552),
 (0.05, 1000, 0.9120000004768372),
 (0.05, 1500, 0.9079999923706055),
 (0.01, 500, 0.8999999761581421),
 (0.01, 1000, 0.8999999761581421),
 (0.01, 1500, 0.9240000247955322), <- Best
 (0.005, 500, 0.8759999871253967),
 (0.005, 1000, 0.8840000033378601),
 (0.005, 1500, 0.9120000004768372),
 (0.001, 500, 0.8840000033378601),
 (0.001, 1000, 0.8799999952316284),
 (0.001, 1500, 0.8960000276565552)]
"""

# Training the model
model_conv.train_multiple_models(learning_rates_conv, epoch_values_conv, print_output=True, predictions=5, plot_graph=False)

# Build and summarize the model
model_conn_1 = ConnectedNN_1_layers(num_classes=50, num_samples_train=15, num_samples_test=5, seed=1, is_colab=True)
model_conn_2 = ConnectedNN_2_layers(num_classes=50, num_samples_train=15, num_samples_test=5, seed=1, is_colab=True)
model_conn_3 = ConnectedNN_3_layers(num_classes=50, num_samples_train=15, num_samples_test=5, seed=1, is_colab=True)

# Define ranges for ConnectedNN
learning_rates_conn = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
epoch_values_conn = [500, 1000, 2000]

# Train the model
result1 = model_conn_1.train_multiple_models(learning_rates_conn, epoch_values_conn, print_output=True, predictions=5, plot_graph=False)
result2 = model_conn_2.train_multiple_models(learning_rates_conn, epoch_values_conn, print_output=True, predictions=5, plot_graph=False)
result3 = model_conn_3.train_multiple_models(learning_rates_conn, epoch_values_conn, print_output=True, predictions=5, plot_graph=False)

"""
With 1 Dense layer
[(1.0, 500, 0.11599999666213989),
 (1.0, 1000, 0.04399999976158142),
 (1.0, 2000, 0.07599999755620956),
 (0.1, 500, 0.1080000028014183),
 (0.1, 1000, 0.1120000034570694),
 (0.1, 2000, 0.10000000149011612),
 (0.01, 500, 0.4880000054836273),
 (0.01, 1000, 0.5320000052452087),
 (0.01, 2000, 0.5479999780654907),
 (0.001, 500, 0.4880000054836273),
 (0.001, 1000, 0.47600001096725464),
 (0.001, 2000, 0.5239999890327454),
 (0.0001, 500, 0.5199999809265137),
 (0.0001, 1000, 0.5),
 (0.0001, 2000, 0.5199999809265137)]

With 2 Dense layers
[(1.0, 500, 0.04399999976158142),
 (1.0, 1000, 0.052000001072883606),
 (1.0, 2000, 0.03200000151991844),
 (0.1, 500, 0.035999998450279236),
 (0.1, 1000, 0.05999999865889549),
 (0.1, 2000, 0.019999999552965164),
 (0.01, 500, 0.5239999890327454),
 (0.01, 1000, 0.5239999890327454),
 (0.01, 2000, 0.5320000052452087),
 (0.001, 500, 0.5120000243186951),
 (0.001, 1000, 0.4959999918937683),
 (0.001, 2000, 0.5040000081062317),
 (0.0001, 500, 0.4959999918937683),
 (0.0001, 1000, 0.527999997138977),
 (0.0001, 2000, 0.4880000054836273)]

With 3 Dense layers
[(1.0, 500, 0.019999999552965164),
 (1.0, 1000, 0.019999999552965164),
 (1.0, 2000, 0.019999999552965164),
 (0.1, 500, 0.019999999552965164),
 (0.1, 1000, 0.019999999552965164),
 (0.1, 2000, 0.019999999552965164),
 (0.01, 500, 0.13600000739097595),
 (0.01, 1000, 0.15199999511241913),
 (0.01, 2000, 0.024000000208616257),
 (0.001, 500, 0.4480000138282776),
 (0.001, 1000, 0.5040000081062317),
 (0.001, 2000, 0.515999972820282),
 (0.0001, 500, 0.492000013589859),
 (0.0001, 1000, 0.47200000286102295),
 (0.0001, 2000, 0.47200000286102295)]

"""
print("With 1 Dense Layer")
print(result1)
print("With 2 Dense Layers")
print(result2)
print("With 3 Dense Layers")
print(result3)

# ----- Experimentation ----- 

# Data Split
model_conn_20_80 = ConnectedNN_1_layers(num_classes=50, num_samples_train=15, num_samples_test=5, seed=1, is_colab=False)
model_conn_50_50 = ConnectedNN_1_layers(num_classes=50, num_samples_train=10, num_samples_test=10, seed=1, is_colab=False)
model_conn_80_20 = ConnectedNN_1_layers(num_classes=50, num_samples_train=16, num_samples_test=4, seed=1, is_colab=False)
model_conn_90_10 = ConnectedNN_1_layers(num_classes=50, num_samples_train=18, num_samples_test=2, seed=1, is_colab=False)

best_lr     = 0.01
best_epoch  = 2000

result_20_80 = model_conn_20_80.train_model(best_epoch, best_lr)
result_50_50 = model_conn_50_50.train_model(best_epoch, best_lr)
result_80_20 = model_conn_80_20.train_model(best_epoch, best_lr)
result_90_10 = model_conn_90_10.train_model(best_epoch, best_lr)

names = ["20/80 Split", "50/50 Split", "80/20 Split", "90/10 Split"]

"""
20/80 Split
0.5640000104904175
50/50 Split
0.4320000112056732
80/20 Split
0.5199999809265137
90/10 Split
0.5600000023841858
"""

for i, item in enumerate([result_20_80, result_50_50, result_80_20, result_90_10]):
  print(names[i])
  print(item.history['val_sparse_categorical_accuracy'][-1])

# %%
model_conv_20_80 = ConvNN(num_classes=50, num_samples_train=15, num_samples_test=5, seed=1, is_colab=False)
model_conv_50_50 = ConvNN(num_classes=50, num_samples_train=10, num_samples_test=10, seed=1, is_colab=False)
model_conv_80_20 = ConvNN(num_classes=50, num_samples_train=16, num_samples_test=4, seed=1, is_colab=False)
model_conv_90_10 = ConvNN(num_classes=50, num_samples_train=18, num_samples_test=2, seed=1, is_colab=False)

best_lr     = 0.01
best_epoch  = 2000

result_20_80 = model_conv_20_80.train_model(best_epoch, best_lr)
result_50_50 = model_conv_50_50.train_model(best_epoch, best_lr)
result_80_20 = model_conv_80_20.train_model(best_epoch, best_lr)
result_90_10 = model_conv_90_10.train_model(best_epoch, best_lr)

names = ["20/80 Split", "50/50 Split", "80/20 Split", "90/10 Split"]

"""
20/80 Split
0.8799999952316284
50/50 Split
0.8059999942779541
80/20 Split
0.8399999737739563
90/10 Split
0.8700000047683716
"""

for i, item in enumerate([result_20_80, result_50_50, result_80_20, result_90_10]):
  print(names[i])
  print(item.history['val_sparse_categorical_accuracy'][-1])


# Data Size
model_conn_10 = ConnectedNN_1_layers(num_classes=10, num_samples_train=15, num_samples_test=5, seed=1, is_colab=False)
model_conn_25 = ConnectedNN_1_layers(num_classes=25, num_samples_train=15, num_samples_test=5, seed=1, is_colab=False)
model_conn_75 = ConnectedNN_1_layers(num_classes=75, num_samples_train=15, num_samples_test=5, seed=1, is_colab=False)

best_lr     = 0.01
best_epoch  = 2000

result_10 = model_conn_10.train_model(best_epoch, best_lr)
result_25 = model_conn_25.train_model(best_epoch, best_lr)
result_75 = model_conn_75.train_model(best_epoch, best_lr)

names = ["10 Categories", "25 Categories", "75 Categories"]

"""
10 Categories
0.6399999856948853
25 Categories
0.4880000054836273
75 Categories
0.4320000112056732
"""

for i, item in enumerate([result_10, result_25, result_75]):
  print(names[i])
  print(item.history['val_sparse_categorical_accuracy'][-1])

# %%
model_conv_10 = ConvNN(num_classes=10, num_samples_train=15, num_samples_test=5, seed=1, is_colab=False)
model_conv_25 = ConvNN(num_classes=25, num_samples_train=15, num_samples_test=5, seed=1, is_colab=False)
model_conv_75 = ConvNN(num_classes=75, num_samples_train=15, num_samples_test=5, seed=1, is_colab=False)

best_lr     = 0.01
best_epoch  = 2000

result_10 = model_conv_10.train_model(best_epoch, best_lr)
result_25 = model_conv_25.train_model(best_epoch, best_lr)
result_75 = model_conv_75.train_model(best_epoch, best_lr)

names = ["10 Categories", "25 Categories", "75 Categories"]

"""
10 Categories
0.8999999761581421
25 Categories
0.8880000114440918
75 Categories
0.8799999952316284
"""

for i, item in enumerate([result_10, result_25, result_75]):
  print(names[i])
  print(item.history['val_sparse_categorical_accuracy'][-1])


