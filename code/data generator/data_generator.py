# This code was made on Google Colab


import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# # # Path to source immage folder
image_path = '/content/drive/MyDrive/Colab Notebooks/Pova Project/Pova Project 2 /Gen Data/src data/scarlett'


# # # Data Genarator COnfiguration
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2)


# # # Import image
train_generator = datagen.flow_from_directory(
    image_path,
    target_size=(200, 200),
    batch_size=1,
    class_mode='binary')


# # # Change train_generator to narray
nb_data = 275  # nb image
data = np.zeros((nb_data, 200, 200, 3))
i = 0
for img, lab in train_generator:
    data[i, :, :, :] = img[0, :, :, :]
    i += 1
    if i >= nb_data:
        break

# # # Save image
save_here = "/content/drive/MyDrive/Colab Notebooks/Pova Project/Pova Project 2 /Gen Data/gen data/scarlett"
datagen.fit(data)
for x, val in zip(datagen.flow(data,
                               save_to_dir=save_here,
                               save_prefix='s2',
                               save_format='png'), range(10)):
    pass
