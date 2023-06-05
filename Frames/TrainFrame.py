import os
import cv2
import customtkinter as ctk
import numpy as np
from IA import MATrain

EMOTIONS_DICT = {
  'Felicidad': 0, 'Tristeza': 1,
  'Ira': 2,       'Sorpresa': 3,
  'Miedo': 4,     'Disgusto': 5,
  'Verguenza': 6, 'Desprecio': 7,
  'Diversion': 8, 'Preocupacion': 9
}

EMOTIONS = [
  'Felicidad', 'Tristeza',
  'Ira',       'Sorpresa',
  'Miedo',     'Disgusto',
  'Verguenza', 'Desprecio',
  'Diversion', 'Preocupacion'
]

class TrainFrame(ctk.CTkFrame):
  def __init__(self, master, **kwargs):
      super().__init__(master, **kwargs)
      
      self.label = ctk.CTkLabel(self, text="Opciones para el entrenamiento")
      self.label.place(relx=0.5, rely=0.05, anchor="center")
      
      self.emotionsLabel = ctk.CTkLabel(self, text="Seleccione la emoción detectada")
      self.emotionsLabel.place(relx=0.1, rely=0.1)
      
      self.emotionOptions = ctk.CTkComboBox(self, values=EMOTIONS, state="readonly")
      self.emotionOptions.place(relx=0.5, rely=0.1)
      self.emotionOptions.set('Felicidad')
      
      self.btnCap = ctk.CTkButton(self, text="Grabar emoción", command=self.captureClick)
      self.btnCap.place(relx=0.23, rely=0.25)
      
      self.btnTrain = ctk.CTkButton(self, text="Entrenar inteligencia", command=self.trainMAInteligence)
      self.btnTrain.place(relx=0.53, rely=0.25)
      
      self.lblState = ctk.CTkLabel(self, text="Estado: Sin grabar")
      self.lblState.place(relx=0.4, rely=0.5)
    
  def readTrainData(self):
      dataPath = 'Data/Train'
      trainImages = []
      imagesTags = []
      
      seed = np.random.randint(0, 30000)
      
      for file in os.listdir(dataPath):
        img = cv2.imread(os.path.join(dataPath, file))
        trainImages.append(img)
        
        tag = file.split(' ')[0].split('@')[1]
        imagesTags.append(tag)
        #print(f"File: {file} Tag: {tag}")
        
      data = np.array(trainImages, dtype=np.float32)
      tags = []
      for tag in imagesTags:
        tags.append(EMOTIONS_DICT[tag])
      
      tags = np.array(tags, dtype=np.int32)
      
      np.random.seed(seed)
      np.random.shuffle(data)
      
      np.random.seed(seed)
      np.random.shuffle(tags)
      
      print(tags)
      
      return data, tags
      
  def trainMAInteligence(self):
      IA = MATrain.TrainMA()
      #print(IA.model.layers)
      X, Y  = self.readTrainData()
      IA.fit(100, X, Y)
      
      #print(f"XShape: {X.shape} YShape: {Y.shape}")
      
  """ def emotionOption(self, choice):
      print("CBox clicked: ", choice) """
      
  def captureClick(self):
    if not self.master.isCapturing:
      self.btnCap.configure(state="disabled")
      self.lblState.configure(text="Estado: grabando...")
      
      #print(f"frames: {np.array(self.master.dataFrames, dtype=np.float32)}")
      self.master.isCapturing = True
    else:
      if len(self.master.dataFrames) > 0:
        self.master.saveData(self.emotionOptions.get())
      
      self.btnCap.configure(state="normal")
      self.lblState.configure(text="Estado: Sin grabar")
      self.master.isCapturing = False  
    