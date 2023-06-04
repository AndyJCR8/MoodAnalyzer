import os
import cv2
import customtkinter as ctk
import numpy as np
from IA import MATest

EMOTIONS = [
  'Felicidad', 'Tristeza',
  'Ira',       'Sorpresa',
  'Miedo',     'Disgusto',
  'Vergüenza', 'Desprecio',
  'Diversión', 'Preocupación'
]

class TestFrame(ctk.CTkFrame):
  def __init__(self, master, **kwargs):
      super().__init__(master, **kwargs)
      
      self.label = ctk.CTkLabel(self, text="Detección de emociones")
      self.label.place(relx=0.5, rely=0.05, anchor="center")

      self.btnTest = ctk.CTkButton(self, command=self.btnTestClick, text="Testear frame local")
      self.btnTest.place(relx=0.5, rely=0.5, anchor="center")
  
  def btnTestClick(self):

      MAModel = MATest.TestMA()
      images = []
      names = []
      
      for file in os.listdir('Data/Test'):
        img = cv2.imread(os.path.join('Data/Test', file))
        images.append(img)
        names.append(file)
      
      images = np.array(images, dtype=np.float32)
      
      index = np.random.randint(0, len(images))
      
      frame = images[index]
      frame = frame[np.newaxis, ...]
      
      #print(f"frame: {frame} // frameShape: {frame.shape}")
      print(f"FrameName: {names[index]} result: {EMOTIONS[MAModel.testFrame(frame)]}")