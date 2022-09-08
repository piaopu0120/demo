import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../src/shape_predictor_68_face_landmarks.dat")

def ldmk_68_detecter(img):
    rects = detector(img)
    sp = predictor(img, rects[0])
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    return landmarks