import tensorflow as tf
import numpy as np
import os
import random
from numpy import array

from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import resize

IMAGE_SIZE = 64
# CHARSET_SIZE = 3755
CHARSET_SIZE = 37

def input(dataset):
    return dataset.images, dataset.labels

class DataSetLoader:
    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%05d' % CHARSET_SIZE)
        print(truncate_path)
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        random.shuffle(self.image_names)
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]
        self.images_rgb = [imread(file_name) for file_name in self.image_names]
        self.image_resized = [resize(image, (IMAGE_SIZE, IMAGE_SIZE)) for image in self.images_rgb]
        self.images = [rgb2gray(item) for item in self.image_resized]

        # convert list to numpy array
        self.images = array(self.images)
        self.labels = array(self.labels)

    @property
    def size(self):
        return len(self.labels)

train_data = DataSetLoader(data_dir='../data/train_/')
test_data = DataSetLoader(data_dir='../data/test_/')
print ('Data loaded ...')



# Specify feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[IMAGE_SIZE, IMAGE_SIZE])]

# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[4096, 256, 256, 256, 64],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=CHARSET_SIZE,
    dropout=0.1,
    model_dir="../dfs/checkpoint/dnn5_model"
)

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(test_data)[0]},
    y=input(test_data)[1],
    num_epochs=None,
    batch_size=50,
    shuffle=True
)
print ('Begin to train ...')

classifier.train(input_fn=train_input_fn, steps=20000)
print ('Train done ...')

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(test_data)[0]},
    y=input(test_data)[1],
    num_epochs=1,
    shuffle=False
)

# Evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
