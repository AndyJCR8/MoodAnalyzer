import os
import tensorflow as tf
import numpy as np
from . import ModelMA

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

class TestMA():
  
  def __init__(self):
    self.model = ModelMA.Model()
    
    try:
      #self.model = tf.keras.models.load_model(os.path.join(CURRENT_DIR, '../model/MAModel.h5'))
      # Llamar al método `call()` del modelo para crear las variables
      dummy_input = tf.zeros((1, 128, 128, 3))  # Ejemplo de entrada dummy
      _ = self.model(dummy_input)
      self.model.load_weights(os.path.join(CURRENT_DIR, '../model/MAModel.h5'))
      
    except Exception as e:
      print(f"Error: {e}")
      
    self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
  
  def testFrame(self, frame):
    res = self.model(frame).numpy()[0].argmax()
    return res 
    
  