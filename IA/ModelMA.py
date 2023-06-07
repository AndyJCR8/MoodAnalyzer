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
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(5, activation='softmax')
    ]

    self.lossf = tf.keras.losses.SparseCategoricalCrossentropy()
    self.opt = tf.keras.optimizers.Adam(0.001)
  
  def call(self, x):
    for i in range(len(self.modelLayers)):
      x = self.modelLayers[i](x)
    return x
  
  @tf.function(jit_compile=False)
  def fitstep(self, X, Y):
      with tf.GradientTape() as tape:
        out = self(X)
        loss = self.lossf(Y, out)
      
      g = tape.gradient(loss, self.trainable_variables)
      self.opt.apply_gradients(zip(g, self.trainable_variables))
      return loss
      
