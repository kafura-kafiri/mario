import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import PIL
import imageio
import cv2
from IPython import display

if not os.path.exists('images'):
    os.system('7z x mario.7z.001')
    os.mkdir('images')
    os.system('ffmpeg -i mario.mp4 -vf fps=10 -s 128x120 images/%d.png')

head, tail = (135, 23108)
height, width = 120, 128

if not os.path.exists('mario.npy'):
    images = np.zeros((tail - head, height, width))
    for i in range(head, tail):
        img = cv2.imread(f'mario/{i}.png', cv2.IMREAD_GRAYSCALE)
        images[i - head] = img

    np.save('mario.npy', images.astype(np.float32))

train_images = np.load('mario.npy')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
train_images = train_images.reshape((tail - head, height, width, 1))

BUFFER_SIZE = tail - head
BATCH_SIZE = 256


def generator():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(int(height / 8) * int(width / 8) * 256, use_bias=False, input_shape=(100, )),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((int(height / 8), int(width / 8), 256)),
        tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])

    assert model.output_shape == (None, height, width, 1)
    return model


a_generator = generator()
noise = tf.random.normal([1, 100])
generated_image = a_generator(noise, training=False)
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()


def discriminator():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(height, width, 1)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

    assert model.output_shape == (None, 1)
    return model


a_discriminator = discriminator()
# print(a_discriminator(generated_image, training=False))
# print(a_discriminator(train_images[: 1].reshape(1, height, width, 1)))


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './mario_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=a_generator,
                                 discriminator=a_discriminator)


EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = a_generator(noise, training=True)

        real_output = a_discriminator(images, training=True)
        fake_output = a_discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, a_generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, a_discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, a_generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, a_discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(a_generator,
                                 epoch + 1,
                                 seed)

        # Save the model
        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    display.clear_output(wait=True)
    generate_and_save_images(a_generator, epochs, seed)


# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

anim_file = 'dcgan.gif'
import glob

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
          last = frame
        else:
          continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)


import IPython
if IPython.version_info > (6, 2, 0, ''):
    display.Image(filename=anim_file)
