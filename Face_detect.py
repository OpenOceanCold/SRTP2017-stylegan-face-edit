# Thanks to StyleGAN2 provider —— Copyright (c) 2019, NVIDIA CORPORATION.
# This work is trained by Copyright(c) 2018, seeprettyface.com, BUPT_GWY.
import os
import bz2
import sys
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
from PIL import Image
import numpy as np


LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

if __name__ == '__main__':

    img_path = sys.argv[1]
    
    file_path = os.path.splitext(img_path)[0]

    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    landmarks_detector = LandmarksDetector(landmarks_model_path)

    face_img_path = file_path + '_face.png'
    for i, face_landmarks in (enumerate(landmarks_detector.get_landmarks(img_path), start=1)):
        image_align(img_path, face_img_path, face_landmarks)
    
    