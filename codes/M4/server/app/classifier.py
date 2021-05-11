from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import numpy as np
import os.path
from django.conf import settings
import cv2


class classifier:
    def __init__(self):
        #Dataset 1 predictive models
        json_file = open('ml/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights("ml/model.h5")

    def emotion_analysis(self, emotions):
        emotion_lib = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        print("Predicted Emotion : ", emotion_lib[int(np.argmax(emotions))])
        return (emotion_lib[np.argmax(emotions)], int(np.argmax(emotions)))


    def make_prediction(self, f):
        f = os.path.abspath(os.path.dirname(__file__))+ '/static/images/'+f
        test_image=cv2.imread(f)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        x, y, face, crop=detect_face(cascade, test_image)
        detected_failed = -1
        if len(crop) is 0:
            return ('No Face Detected - Retry.', detected_failed)
        else:
            emotion_pred=self.emotion_analysis(self.loaded_model.predict(crop[0]))
            return (emotion_pred)

def detect_face(cascade, pic, scaleFactor=1.3):
  #incase we change the original one
  img=pic.copy()
  #convert image into gray scale as opencv face detector expects gray images
  gray_image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #use haar classifier to detect faces
  face_box=cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
  cropped=[]
  xs=[]
  ys=[]
  for(x,y,w,h) in face_box:
    cv2.rectangle(img, (x,y), (x+w, y+h+10), (0,255,0), 2)
    #crop the boxed face
    gray_frame=gray_image[y:y+h,x:x+w]
    cropped_img = cv2.resize(gray_frame, (48,48))
    cropped_img = image.img_to_array(cropped_img)
    cropped_img = cv2.resize(cropped_img.astype('uint8'), (64,64))
    cropped_img = cropped_img.astype('float32')
    cropped_img = ((cropped_img / 255.0) - 0.5) * 2.0
    cropped_img = np.expand_dims(cropped_img, 0)
    cropped_img = np.expand_dims(cropped_img,-1)
    cropped.append(cropped_img)
    xs.append(x)
    ys.append(y)
    
  return xs, ys, img, cropped