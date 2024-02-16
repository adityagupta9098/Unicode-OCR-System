import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk




# def main():

    # Set the path to your dataset
dataset_path = r"F:\data\training_data"

# Define constants
img_size = (28, 28)  # Adjust the size based on your dataset
batch_size = 32

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(img_size[0], img_size[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the trained model
model.save('OCR_model.h5')

# Function for inference


def predict_character(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Convert class index to character
    characters = {v: k for k, v in train_generator.class_indices.items()}
    predicted_character = characters[predicted_class]

    return predicted_character









# Example usage
image_path = r"F:\data\testing_data\7\28425.png"
predicted_char = predict_character(image_path)
print(f"Predicted character: {predicted_char}")
