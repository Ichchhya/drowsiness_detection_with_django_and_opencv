import cv2
import os
import numpy as np
from keras import models
from django.conf import settings
from playsound import playsound


# face = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'haarcascade_frontalface_alt.xml'))
# leye = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'haarcascade_lefteye_2splits.xml'))
# reye = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'haarcascade_righteye_2splits.xml'))
#

face = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'haarcascade_frontalface_alt.xml'))
leye = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'haarcascade_lefteye_2splits.xml'))
reye = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'haarcascade_righteye_2splits.xml'))

model = models.load_model(os.path.join(settings.BASE_DIR, 'face_detector/cnnCat2.h5'))


class VideoCamera(object):
    lbl = ['Close', 'Open']
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count = 0
    score = 0
    thick = 2
    rpred = [99]
    lpred = [99]

    def __init__(self):
        self.cap = cv2.VideoCapture("test.flv")
        # self.cap = cv2.VideoCapture(0)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self,score=0):
        ret, frame = self.cap.read()
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)
        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            count = self.count + 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            self.rpred = model.predict_classes(r_eye)
            if self.rpred[0] == 1:
                lbl = 'Open'
            elif self.rpred[0] == 0:
                lbl = 'Closed'
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            count = self.count + 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            self.lpred = model.predict_classes(l_eye)
            if (self.lpred[0] == 1):
                lbl = 'Open'
            if (self.lpred[0] == 0):
                lbl = 'Closed'
            break

        if (self.rpred[0] == 0 and self.lpred[0] == 0):
            score = self.score + 1
            cv2.putText(frame, "Closed", (10, height - 20), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            # if(rpred[0]==1 or lpred[0]==1):
        else:
            score = self.score - 1
            cv2.putText(frame, "Open", (10, height - 20), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score < 0:
            score = 0
        cv2.putText(frame, 'Score:' + str(score), (100, height - 20), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if score > 15:
            # person is feeling sleepy so we beep the alarm

            try:
                playsound(os.path.join(settings.BASE_DIR, 'alarm.wav'))
            except:  # isplaying = False
                pass
            if self.thick < 16:
                thick = self.thick + 2
            else:
                thick = self.thick - 2
                if thick < 2:
                    thick = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thick)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

