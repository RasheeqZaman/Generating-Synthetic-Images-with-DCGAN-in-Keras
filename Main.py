import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
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
#plot_image(generated_images)

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

discriminator.compile(loss='binary_crossentropy', optimizer='rmsprop')
discriminator.trainable = False
gan = tf.keras.models.Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer='rmsprop')


seed = tf.random.normal(shape=[batch_size, num_features])

def train_dcgan(gan, dataset, batch_size, num_features, epochs=5):
    generator, discriminator = gan.layers
    for epoch in tqdm(range(epochs)):
        print("Epoch: {}/{}".format(epoch+1, epochs))
        for X_batch in dataset:
            noise = tf.random.normal(shape=[batch_size, num_features])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]]*batch_size + [[1.]]*batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            y2 = tf.constant([[1.]]*batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
        generate_and_save_images(generator, epoch+1, seed)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    show_images(predictions)
