"""
Face detection
"""
import os

# Imports
import cv2
import dlib
import imutils
import keras
import tensorflow as tf
import numpy as np
from imutils.face_utils import FaceAligner, rect_to_bb
from keras.layers import Dense
from keras.optimizers import Adam

from wide_resnet import WideResNet


class FaceCV(object):
    """
    Singleton class for face recongnition task
    """
    CASE_PATH = "https://github.com/Tony607/Keras_age_gender/blob/master/pretrained_models/haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=256):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.path.dirname(__file__), './models/')
        fpath = tf.keras.utils.get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)
        self.model.layers.pop()
        self.model.layers.pop()
        output_layer = self.model.layers[-1].output
        x_newfc = Dense(70, activation='softmax', name='classification')(output_layer)
        new_model = keras.Model(inputs=self.model.input, outputs=x_newfc)
        adam = Adam()
        new_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', 'crossentropy'])
        new_model.load_weights("./models/asian_weights.hdf5")
        self.model = new_model

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin )
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self, image):
        # Create a face detector
        detector = dlib.get_frontal_face_detector()
        # Create a face aligner
        predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
        fa = FaceAligner(predictor, desiredFaceHeight=64, desiredFaceWidth=64)
        # Get image directory
        image_name = image
        # Read, resize image and change to greyscale
        image = cv2.imread(image)
        image = imutils.resize(image, width=256)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Cropped and aligned faces
        face_imgs = np.empty((1, self.face_size, self.face_size, 3))
        faces = detector(gray, 2)

        (x, y, w, h) = rect_to_bb(faces[0])
        faceOrig = imutils.resize(image[y:y + h, x:x + w], height=64, width=64)
        faceAligned = fa.align(image, gray, faces[0])

        face_imgs[0, :, :, :] = faceAligned

        # Get results
        results = self.model.predict(face_imgs)
        ages = np.arange(0, 70).reshape(70, 1)
        predicted_ages = results[0].dot(ages).flatten()
        index = int(round(predicted_ages[0], 0))
        return index