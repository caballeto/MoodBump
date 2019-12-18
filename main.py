import cv2
import time
import config
import os
import face_recognition

from recognize import identify_faces
from emotions import predict_emotions
from stats import to_html

# 1. Run video capture webcam.
# 2. Extract frame.
# 3. Capture all faces in image.
# 4. Identify faces.
# 5. For all identified faces predict their emotions.
# 6. Append the data into json, (face_name -> (date_time -> (emotion : rate)))

# 7. Create set of api methods to extract statistics.
# 8. Create EEL wrapper for each stats function.
# 9. When stats function is called, forward its request from eel wrapper to handler,
#    then extract stats, compute and return.

millis_time = lambda: int(round(time.time() * 1000))

emotion_records = {}

known_face_names = []
known_face_encodings = []


def load_known_people():
  global known_face_names
  known_face_names = list(os.listdir(config.IMAGE_DIR))

  for face_name in known_face_names:
    face_img = face_recognition.load_image_file(config.IMAGE_DIR + "/" + face_name)
    face_enc = face_recognition.face_encodings(face_img)[0]
    known_face_encodings.append(face_enc)

  known_face_names = list(map(lambda x: os.path.splitext(x)[0], known_face_names))


def main():
  load_known_people()
  process_this_frame = True
  cap = cv2.VideoCapture(0)

  face_locations = []
  face_names = []

  while True:
    ret, image = cap.read()

    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    if process_this_frame:
      face_locations, face_names = identify_faces(small_frame, known_face_names, known_face_encodings)
      emotions = predict_emotions(face_locations, face_names, small_frame)

      # ("employee_name" : [{"angry":, ..., "time": 123}])

      curr_time = millis_time()
      for name in emotions.keys():
        emotions[name]['time'] = curr_time
        if name in emotion_records:
          emotion_records[name].append(emotions[name])
        else:
          emotion_records[name] = [emotions[name]]


    process_this_frame = not process_this_frame

    ## DRAW RESULT
    for (top, right, bottom, left), name in zip(face_locations, face_names):
      top *= 4
      right *= 4
      bottom *= 4
      left *= 4
      # Draw a box around the face
      cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

      # Draw a label with a name below the face
      cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
      font = cv2.FONT_HERSHEY_DUPLEX
      cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
      break

  print(emotion_records)

  to_html(emotion_records)

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
