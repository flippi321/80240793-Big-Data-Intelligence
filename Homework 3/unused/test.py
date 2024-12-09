from utils import *

train_image, train_label, test_image, test_label = LoadData(num_classes = 50, num_samples_per_class_train = 15, num_samples_per_class_test = 5, seed = 1)

print(train_image.shape)
print(train_label.shape)

print(train_image[0])
array_to_image_file(train_image[0], "output_image.png", [28, 28])
print(train_label[0])