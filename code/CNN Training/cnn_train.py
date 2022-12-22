# This code was made on Google Colab


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Rescaling, BatchNormalization
from tensorflow.keras.models import Model

from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

from tensorflow.keras.preprocessing import image_dataset_from_directory


# # # Get Training and Validation dataset

def get_train_and_validation_data(path, validation_split, img_size, batch_size):
    # training data
    train_data = image_dataset_from_directory(
        path,
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale")

    # validation data
    validation_data = image_dataset_from_directory(
        path,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale")

    return train_data, validation_data


# 2_class data_dir = "../data/2_class"
# 100_class data_sir = "../data/100_class"
data_dir = '/content/drive/MyDrive/Colab Notebooks/Pova Project/Pova Project 2 /data'
batch_size = 128
input_size = (200, 200, 1)
validation_split = 0.2  # train split = 0.8
train_data, validation_data = get_train_and_validation_data(
    data_dir, validation_split, input_size[:2], batch_size)

# # # Show Imges and its Labels

class_names = train_data.class_names
max_size = validation_data.cardinality().numpy()  # nb of batch

col = 3
row = 3

nb_img = col*row

plt.figure(figsize=(5, 5))

for i in range(nb_img):
    take_id = np.random.randint(0, max_size)  # random batch id
    for images, labels in train_data.take(take_id):
        ax = plt.subplot(row, col, i+1)
        img = images[0].numpy().astype("uint8")[:, :, 0]
        plt.imshow(img,  cmap='gray', vmin=0, vmax=255)
        plt.title(class_names[labels[0]], fontsize=10)
        plt.axis("off")


# # # Model Build
# This is just an example of layers used to train a model

def build_CNN(input_shape, output_size):

    # Don't Touch ------------------------------------
    inputs = Input(shape=input_shape, name="input")
    rescal = Rescaling(scale=1/255)(inputs)
    # -------------------------------------------------

    # Change this part
    # Convolution Layers
    # Parameters :
    #   * filters = [8, 16, 32, 64, 128, 256]
    #   * kernel_size = [3,5]
    #   * activation = https://keras.io/api/layers/activations/
    conv2d_1 = Conv2D(filters=32, kernel_size=5, activation='relu')(rescal)
    norm_1 = BatchNormalization()(conv2d_1)
    maxPool_1 = MaxPooling2D(pool_size=(2, 2))(
        norm_1)

    flatten = Flatten()(maxPool_1)

    # Change this part
    # Neural Layers
    # Parameters :
    #   * nb of neurones for Dense  = randomm int
    #   * dropout_value = float between 0. - 1.
    # Info : neural layer = Dense or Dense + Dropout
    dense_1 = Dense(256, activation='relu')(flatten)
    dropout_1 = Dropout(0.15)(dense_1)

    # Ouput
    # Juste change the link name with previous layer
    outputs = Dense(output_size, activation='softmax')(dropout_1)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


# Change this part
# Parameter :
#   * learning rate  = smal number, example 1e-3 or 1-4 or 1e-2
#   * max_epoch = randomm int
learning_rate = 1e-3
max_epoch = 50

output_size = len(validation_data.class_names)
model = build_CNN(input_size, output_size)

# # # Train the model

model.compile(
    loss=losses.SparseCategoricalCrossentropy(),
    optimizer=optimizers.Adam(learning_rate),
    metrics=[metrics.sparse_categorical_accuracy])


history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=max_epoch
)

# # # Plot Losses and Accuraccy

train_loss = history.history["loss"]
train_accuracy = history.history["sparse_categorical_accuracy"]
val_loss = history.history["val_loss"]
val_accuracy = history.history["val_sparse_categorical_accuracy"]
epoch = [e for e in range(1, len(train_loss)+1)]

# Losses
plt.figure(figsize=(15, 5))
ax = plt.subplot(1, 2, 1)
plt.plot(epoch, train_loss, "r", label="train")
plt.plot(epoch, val_loss, "b", label="validation")
plt.title("Loss")
plt.xlabel("epoch")
plt.legend()

# Accuracy
ax = plt.subplot(1, 2, 2)
plt.plot(epoch, train_accuracy, "r", label="train")
plt.plot(epoch, val_accuracy, "b", label="validation")
plt.title("Acurracy")
plt.xlabel("epoch")
plt.legend()

plt.show()

print("max acc : ", max(val_accuracy))
print("epoch : ", np.argmax(np.array(val_accuracy)))
print("mean acc : ", np.average(np.array(val_accuracy)))

# # # Show Predict Result


class_names = train_data.class_names
max_size = validation_data.cardinality().numpy()  # nb of batch

col = 3
row = 3

nb_img = col*row

plt.figure(figsize=(5, 5))

for i in range(nb_img):
    take_id = np.random.randint(0, max_size)  # random batch id
    for images, labels in validation_data.take(take_id):
        ax = plt.subplot(row, col, i+1)

        img = images[0].numpy().astype("uint8")
        predict = model.predict(np.array([img.tolist()]), verbose=0)
        arg_max = np.argmax(predict)

        plt.imshow(img[:, :, 0],  cmap='gray', vmin=0, vmax=255)
        plt.title(class_names[arg_max])
        plt.axis("off")


# # # Save inside your Drive
# Choose .h5 extension
model.save('/content/drive/MyDrive/Colab Notebooks/Pova Project/Pova Project 2 /Try 1/kavin/model/84p.h5')
