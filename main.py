import cv2
import customtkinter as ctk
import numpy as np
from typing import Optional, Tuple, Union
from PIL import Image, ImageTk
from datetime import datetime
from Frames import TrainFrame as TRF, MainFrame as MF

EMOTIONS = [
  'Felicidad', 'Tristeza',
  'Ira',       'Miedo',
  'Disgusto'
]

class App(ctk.CTk):
    def __init__(self, title, size):
        super().__init__()
        self.title(title)
        self.geometry(f"{size[0]}x{size[1]}")
        self.minsize(size[0], size[1])
        
        # add widgets to app
        self.lblTitle = ctk.CTkLabel(self, text="Analizador de emociones MoodAnalizer", font=("", 18))
        self.lblTitle.pack()
        
        self.switchVar = ctk.StringVar(value="off")
        self.switchMode = ctk.CTkSwitch(self, text="Modo entrenamiento", variable=self.switchVar, command=self.toggleMode, onvalue="on", offvalue="off")
        self.switchMode.pack()
        
        self.modeFrame = MF.MainFrame(master=self, width=500, height=500)
        self.video = cv2.VideoCapture(0)
        self.detectionColor = (np.random.randint(50, 256), np.random.randint(50, 256), np.random.randint(50, 256))
        
        #print(f"Width: f{self.video.get(cv2.CAP_PROP_FRAME_WIDTH)} Height: f{self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self.canvas = ctk.CTkCanvas(self, width=self.video.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        
        self.dataFrames = []
        self.framesCount = 0
        self.isCapturing = False
        self.isFace = False
        
        self.update_frame()
        self.mainloop()
    
    def setDefValues(self):
        self.dataFrames = []
        self.framesCount = 0
        self.isCapturing = False
        self.isFace = False
    
    def toggleMode(self):
      if self.switchVar.get() == "on":
        self.modeFrame = TRF.TrainFrame(master=self, width=500, height=500)
      else: self.modeFrame = MF.MainFrame(master=self, width=500, height=500)
    
    def detectFace(self, frame):
      face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
      # Convierte el cuadro a escala de grises
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

      # Dibuja un cuadro alrededor de cada rostro detectado
      for (x, y, w, h) in faces:
          cv2.rectangle(frame, (x-5, y-5), ((x+w) + 5, (y+h) + 5), self.detectionColor, 2)
          
      return faces
    
    def saveData(self, emotionTag):
      data = np.array(self.dataFrames, dtype=np.float32)
      
      splitIndex = int(len(data) * 0.8)
      trainFrames, testFrames = np.split(data, [splitIndex])
      
      for i, trainFrame in enumerate(trainFrames):
        cv2.imwrite(f"./Data/Train/{i + 1}@{emotionTag} {datetime.now().strftime('%d-%m-%Y %H_%M_%S')}.jpg", trainFrame)
        
      for i, testFrame in enumerate(testFrames):
        cv2.imwrite(f"./Data/Test/{i + 1}@{emotionTag} {datetime.now().strftime('%d-%m-%Y %H_%M_%S')}.jpg", testFrame)
        
      #print(f"trainframes: {trainFrames} testframes: {testFrames}")
      self.setDefValues()
    
    def captureFrames(self):
      ret, frame = self.video.read()
      frame = cv2.flip(frame, 1)
      faces = self.detectFace(frame)

      #print(f"len: {len(np.shape(faces))}")
      if len(np.shape(faces)) == 2:
        self.isFace = True
      else: self.isFace = False
      
      if ret:
          cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          image = Image.fromarray(cv2image)
          
          self.photo = ImageTk.PhotoImage(image=image)
          self.canvas.create_image(0, 0, anchor=ctk.NW, image=self.photo)
          self.canvas.place(relx=0.28, rely=0.55, anchor="center")
          
          frameCutted = None
          
          if self.isFace:
            faces = faces[0]
            frameCutted = frame[faces[1]: faces[1] + faces[3], faces[0]: faces[0] + faces[2]]
            #print(f"frameCutted: {frameCutted.shape} frame: {frame.shape}")
            #print(f"faces: {faces}")
            
          return frameCutted
          #return frame
        
    def update_frame(self):
        
        self.lblTitle.place(relx=0.5, rely=0.05, anchor="center")
        self.switchMode.place(relx=0.8, rely=0.1, anchor="center")
        self.modeFrame.place(relx=0.77, rely=0.55, anchor="center")
        
        frame = self.captureFrames()
        if self.modeFrame.label._text == 'Opciones para el entrenamiento':
          if frame is not None and self.isFace and self.isCapturing:
            #self.saveData()
            if self.framesCount < 100:
              resizedFrame = cv2.resize(frame, (128, 128))
              self.dataFrames.append(resizedFrame)
              self.framesCount += 1
              
            elif self.framesCount == 100:
              self.modeFrame.captureClick()
              self.framesCount = 0
              
        
        self.after(15, self.update_frame)
    
App("Mood Analizer", (1200, 600))