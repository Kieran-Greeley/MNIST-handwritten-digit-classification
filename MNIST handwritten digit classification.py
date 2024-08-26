import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


# Load the MNIST dataset using TensorFlow
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the images to [-0.5, 0.5]
train_images = (train_images / 255.0) - 0.5
test_images = (test_images / 255.0) - 0.5

# Flatten the images to (batch_size, 784)
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Build the model.
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dropout(0.25),
  Dense(64, activation='relu'),
  Dropout(0.25),
  Dense(10, activation='softmax'),
])

# Compile the model.
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=10,
  batch_size=64,
  validation_data=(test_images, to_categorical(test_labels))
)

# Evaluate the model.
model.evaluate(
  test_images,
  to_categorical(test_labels)
)

# Predict on the first 50 test images.
predictions = model.predict(test_images[:50])

# Check our predictions against the ground truths.
for i in range(37):
    print(f"Image {i+1}: Predicted {np.argmax(predictions[i])}, True {test_labels[i]}")
