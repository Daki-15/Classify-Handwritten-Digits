# MNIST Neural Network with TensorFlow in Google Colab

This notebook demonstrates the process of building, training, and saving a neural network model using TensorFlow and the MNIST dataset. It includes steps to load the dataset, create the model, train it, make predictions, and save/load the model.

## Dependencies

Ensure you have the following libraries installed:
- numpy
- matplotlib
- tensorflow

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from datetime import datetime
```

## Loading the Dataset

The MNIST dataset is loaded and split into training and testing sets.

```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
NUM_IMGAGES = 60000  # Number of images to use for training
```

## Displaying Dataset Images

A helper function to display images from the dataset using Matplotlib.

```python
def display(idx, data_set):
  plt.imshow(data_set[idx])

display(0, X_train)
```

## Creating the Neural Network Model

A function to create a simple neural network model using TensorFlow.

```python
def create_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  print(model.summary())

  return model

model = create_model()
NUM_EPOCHS = 15  # Number of epochs for training
model.fit(X_train[:NUM_IMGAGES], y_train[:NUM_IMGAGES], epochs=NUM_EPOCHS)
```

## Making Predictions

Using the trained model to make predictions on the test set.

```python
predictions = model.predict([X_test])

def prediction(idx):
  print(f'Label -> {y_test[idx]}')
  print(f'Prediction -> {np.argmax(predictions[idx])}')
  display(idx, X_test)
```

## Displaying Multiple Images

A helper function to display multiple images in a grid format.

```python
def display_n_images(images, N, ncolumns=5):
  nrows = (N + ncolumns - 1) // ncolumns
  plt.figure(figsize=(15, nrows * 3,))

  for i in range(N):
    plt.subplot(nrows, ncolumns, i+1)
    plt.title(y_test[i])
    plt.imshow(images[i])

  plt.tight_layout()
  plt.show()
```

## Saving the Model

Function to save the trained model to a specified path.

```python
MODEL_PATH = "/content/drive/MyDrive/ML"

def save_model(model_to_save):
  model_to_save.save(f"{MODEL_PATH}/model--{datetime.now().strftime('%m:%d:%y, %H:%M')}_{NUM_IMGAGES}_images.h5")
```

## Loading the Model

Function to load a saved model from a specified path.

```python
def load_model(MODEL_PATH):
  return tf.keras.models.load_model(MODEL_PATH)
```

## Conclusion

This notebook provides a comprehensive guide to building a neural network model for the MNIST dataset, including steps for loading data, training the model, making predictions, and saving/loading the model. Use the provided functions and modify them as needed for your own experiments and projects.
