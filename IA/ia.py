import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):
  #size: 512x128
  def __init__(self):
    super().__init__()
    
    self.modelLayers = [
      tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu", strides=1),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding="same", activation="relu", strides=1),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding="same", activation="relu", strides=1),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding="same", activation="relu", strides=1),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense()
    ]