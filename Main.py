import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print('Tensorflow Version: ', tf.__version__)

#Loading Images
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

#Normalize Images
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

#Show some images
def show_images(images):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap='gray')
    plt.show()

show_images(train_images)


batch_size = 32
#Shuffle Dataset
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(1000)
#Split into minibatches, prefetches 1 minibatch
dataset = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(1)
