import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam

class Model():
  #size: 512x128
  def __init__(self):
    super().__init__()
    
    self.modelLayers = [
      Conv2D(8, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(128, 128, 3)),
      MaxPooling2D(pool_size=(2, 2)),
      Conv2D(12, kernel_size=(3, 3), padding="same", activation='relu'),
      MaxPooling2D(pool_size=(2, 2)),
      Conv2D(16, kernel_size=(3, 3), padding="same", activation='relu'),
      MaxPooling2D(pool_size=(2, 2)),
      Dropout(0.25),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.5),
      Dense(128, activation='relu'),
      Dense(10, activation='relu')
    ]

def train(frame, emotion, descriptions, model: Sequential):
  
  model()
  pass
