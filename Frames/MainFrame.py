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

class MainFrame(ctk.CTkFrame):
  def __init__(self, master, **kwargs):
      super().__init__(master, **kwargs)
      
      self.label = ctk.CTkLabel(self, text="Ingreso de datos del usuario")
      self.label.place(relx=0.5, rely=0.05, anchor="center")

      """ self.btnTest = ctk.CTkButton(self, command=self.btnTestClick, text="Testear frame local")
      self.btnTest.place(relx=0.5, rely=0.5, anchor="center") """
      
      
      self.txbNombre = ctk.CTkEntry(self, placeholder_text="Nombre de usuario", width=175, height=20, corner_radius=5)
      self.txbNombre.place(relx=0.13, rely=0.15)
      
      self.txbEdad = ctk.CTkEntry(self, placeholder_text="Edad", width=175, height=20, corner_radius=5)
      self.txbEdad.place(relx=0.53, rely=0.15)
      
      self.txbTelefono = ctk.CTkEntry(self, placeholder_text="Teléfono", width=175, height=20, corner_radius=5)
      self.txbTelefono.place(relx=0.13, rely=0.25)
      
      self.txbEmail = ctk.CTkEntry(self, placeholder_text="Correo electrónico", width=175, height=20, corner_radius=5)
      self.txbEmail.place(relx=0.53, rely=0.25)
      
      self.lblSexo = ctk.CTkLabel(self, text="Sexo: ")
      self.lblSexo.place(relx=0.35, rely=0.4)
      
      self.sexCombo = ctk.CTkComboBox(self, values=['Hombre', 'Mujer'], state="readonly")
      self.sexCombo.place(relx=0.45, rely=0.4)
      self.sexCombo.set('Hombre')
      
      self.lblEmocion = ctk.CTkLabel(self, text="Emoción detectada: ")
      self.lblEmocion.place(relx=0.25, rely=0.55)
      
      self.saveData = ctk.CTkButton(self, text="Guardar info", width=175, height=40)
      self.saveData.place(relx=0.13, rely=0.65)
      
      self.detectEmotion = ctk.CTkButton(self, text="Detectar emoción", width=175, height=40)
      self.detectEmotion.place(relx=0.53, rely=0.65)
      
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