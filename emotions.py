import numpy as np
import cv2
import config

from keras.models import model_from_json
from keras.preprocessing import image

model = model_from_json(open(config.MODEL, "r").read())
model.load_weights(config.MODEL_WEIGHTS)

emotion_values = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


def predict_emotions(face_locations, face_names, img):
  emotions = {}

  for (top, right, bottom, left), name in zip(face_locations, face_names):
    detected_face = img[int(top):int(bottom), int(left):int(right)]  # crop detected face
    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
    detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

    img_pixels = image.img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis=0)

    img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

    predictions = model.predict(img_pixels)[0]  # store probabilities of 7 expressions

    emotions[name] = {}
    for index, prediction in enumerate(predictions.tolist()):
      emotions[name][emotion_values[index]] = prediction

    emotion = emotion_values[np.argmax(predictions)]
    print("###################### EMOTION: " + emotion)

  return emotions
