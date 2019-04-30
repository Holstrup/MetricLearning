from MetricLearning.Model import Model
import os
import numpy as np
from keras_preprocessing import image
from tensorflow.python.client import session
import tensorflow as tf


def get_data():
    # image folder
    folder_path = os.getcwd() + '/MetricLearning/Data/'
    img_width, img_height = 224, 224
    images = []
    label = []
    for _, dirs, _ in os.walk(folder_path, topdown=True):
        for directory in dirs:
            sub_folder_path = os.path.join(folder_path, directory)
            for _, _, files in os.walk(sub_folder_path):
                for name in files:
                    if name != '.DS_Store':
                        img = os.path.join(sub_folder_path, name)
                        img = image.load_img(img, target_size=(img_width, img_height))
                        img = image.img_to_array(img)
                        img = np.expand_dims(img, axis=0)
                        images.append(img)
                        label.append(directory)
    new_label = [label.index(x) + 1 for x in label]
    images = np.vstack(images)
    return images, new_label

def run_model():
    images, label = get_data()
    model = Model()
    #label_test = np.arange(0, len(images), 1)

    model.model.fit(x=images, y=label, batch_size=32, epochs=10, verbose=2)


run_model()