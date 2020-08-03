import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

#show_images(train_images)


batch_size = 32
#Shuffle Dataset
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(1000)
#Split into minibatches, prefetches 1 minibatch
dataset = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(1)

def plot_image(gen_image):
    img = np.array(gen_image)
    img = (img[0, :, :, :] + 1.) / 2.
    img = Image.fromarray(img, 'RGB')
    img.show()

num_features = 100
generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(7*7*128, input_shape=[num_features]),
    tf.keras.layers.Reshape([7, 7, 128]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(64, (5, 5), (2, 2), padding='same', activation='selu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(1, (5, 5), (2, 2), padding='same', activation='tanh')
])

noise = tf.random.normal(shape=[1, num_features])
generated_images = generator(noise, training=False)
plot_image(generated_images)

discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (5, 5), (2, 2), padding='same', input_shape=[28, 28, 1]),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, (5, 5), (2, 2), padding='same'),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

decision = discriminator(generated_images)
print(decision)