#!/bin/python
import os

#Image
from PIL import Image
import imghdr

# Math
import matplotlib.pyplot as plt

# Learn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

class Classifier:

    def __init__(self):
        self.image_size = (128, 128)
        self.batch_size = 32
        self.epochs = 50 
        self.gen_dataset()
        self.data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(0.1),
            ]
        )
        self.callbacks = [
            keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")
        ]

        self.model = self.make_model(self.image_size + (3,), 2)
        keras.utils.plot_model(self.model, show_shapes=True)

    def gen_dataset(self):
        self.train_ds = keras.preprocessing.image_dataset_from_directory(
            "PetImages",
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=self.image_size,
            batch_size=self.batch_size
        )
        self.val_ds = keras.preprocessing.image_dataset_from_directory(
            "PetImages",
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=self.image_size,
            batch_size=self.batch_size
        )
        self.test_ds = keras.preprocessing.image_dataset_from_directory(
            "TestImages",
            image_size=self.image_size,
            batch_size=self.batch_size
        )

    def plot_images(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.test_ds.take(1):
            for i in range(32):
                plt.subplot(4, 8, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(int(labels[i]))
                plt.axis("off")
        plt.show()

    def filter_img(self):
        num_skipped = 0

        for folder_name in ("Cat", "Dog"):
            folder_path = os.path.join("PetImages", folder_name)
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)

                try:
                    fobj = open(fpath, "rb")
                    is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
                finally:
                    fobj.close()

                if not is_jfif:
                    num_skipped += 1
                    os.remove(fpath)
        print("Filtered %d images" % num_skipped)

    def make_model(self, input_shape, num_classes):
        inputs = keras.Input(shape=input_shape)

        # Image augmentation block
        x = self.data_augmentation(inputs)

        # Entry block
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        cache_block_activation = x

        for size in [128, 256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                cache_block_activation
            )
            x = layers.add([x, residual])
            cache_block_activation = x

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes

        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(units, activation=activation)(x)

        return keras.Model(inputs, outputs)


    def train(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print(self.train_ds)

        self.model.fit(
            x=self.train_ds,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_data=self.val_ds
        )

    def test(self):
        print("Evaluating model ... ")
        print(self.model.evaluate(x=self.test_ds, verbose=1, return_dict=True))


    def serialize(self):
        self.model.save("modelv0")

def main():
    cl = Classifier()
    cl.train()
    cl.test()
    cl.serialize()

if __name__ == "__main__":
    main()
