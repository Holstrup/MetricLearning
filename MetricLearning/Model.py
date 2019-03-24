import keras
from keras_preprocessing import image
import os
import numpy as np
import database_actions as db_actions


class Model:
    def __init__(self):
        self.model = self.build_model()
        self.compile_model()

    def build_model(self, embedding_size=128):
        base_model = keras.applications.ResNet50(weights='imagenet', include_top=False,
                                                 input_tensor=keras.layers.Input(shape=(224, 224, 3)))
        x = base_model.output
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(embedding_size, activation='softmax', kernel_initializer='glorot_uniform')(x)
        full_model = keras.Model(inputs=base_model.input, outputs=x)
        return full_model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss=keras.losses.MSE,
                           metrics=['accuracy'])



def store_images():
    """
    DO NOT CALL THIS FUNCTION UNLESS YOU HAVE DATA YOU WANT TO STORE
    IT DELETES THE CURRENT DATA.

    """
    # image folder
    folder_path = os.getcwd() + '/Data/'
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

    images = np.vstack(images)
    model = Model()
    predictions = model.model.predict(images, batch_size=20)
    db_actions.reinitialize_table()
    for i in range(100):
        db_actions.add_encoding(predictions[i, :], label[i])



def get_data():
    """
    :return: encodings array of (128, n)
             labels list of (n)
    """
    query = "SELECT * FROM embeddings WHERE label IS NOT NULL"
    cursor, connection = db_actions.connect()
    cursor.execute(query)


    result_list = cursor.fetchall()
    encodings = np.zeros((128, len(result_list)))
    labels = []

    for i in range(len(result_list)):
        encodings[:, i] = result_list[i][0]
        labels.append(result_list[i][1].encode())
    encodings = np.nan_to_num(encodings)
    return encodings.astype('float32'), labels