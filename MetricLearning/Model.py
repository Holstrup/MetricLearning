import keras
from keras_preprocessing import image
import keras.backend as K
import os
import numpy as np
import MetricLearning.database_actions as db_actions
from MetricLearning.triplet_loss import batch_all_triplet_loss
import tensorflow as tf
from tensorflow.python import debug as tf_debug



class Model:
    def __init__(self):
        self.model = self.build_model()
        self.compile_model()

    def build_model(self):
        base_model = keras.applications.ResNet50(weights='imagenet', include_top=False,
                                                 input_tensor=keras.layers.Input(shape=(224, 224, 3)))
        x = base_model.output
        x = keras.layers.AveragePooling2D(pool_size=(7, 7), strides=None, padding='valid')(x)
        x = keras.layers.Softmax()(x)
        full_model = keras.Model(inputs=base_model.input, outputs=x)
        return full_model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss=triplet_loss,
                           metrics=['accuracy'])




def triplet_loss(y_true, y_pred):
    print("Y True (labels) shape: {}".format(K.int_shape(y_true)))
    print("Y Pred (embeddings) shape: {}".format(K.int_shape(y_pred)))

    # Reshape data
    y_pred = K.squeeze(y_pred, axis=1)
    y_pred = K.squeeze(y_pred, axis=1)
    y_true = tf.reshape(y_true, [-1])
    def_margin = tf.constant(1.0, dtype=tf.float32)

    # Print
    #y_true = K.print_tensor(y_true, message='y_true is = ')

    # Run
    loss, _ = batch_all_triplet_loss(embeddings=y_pred, labels=y_true, margin=def_margin)
    return loss


def store_images(database):
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
    db_actions.reinitialize_table(database)
    for i in range(100):
        prediction = predictions[i, :]
        normalized_prediction = prediction / np.sum(prediction)
        db_actions.add_encoding(database, normalized_prediction, label[i])
        print("Sum is: {}".format(np.sum(normalized_prediction)))


def get_data(dataset):
    """
    :return: encodings array of (2048, n)
             labels list of (n)
    """
    query = "SELECT * FROM embeddings WHERE label IS NOT NULL"
    cursor, connection = db_actions.connect(dataset)
    cursor.execute(query)


    result_list = cursor.fetchall()
    encodings = np.zeros((2048, len(result_list)))
    labels = []

    for i in range(len(result_list)):
        encodings[:, i] = result_list[i][0]
        labels.append(result_list[i][1].encode())
    encodings = np.nan_to_num(encodings)
    labels = [x.decode('utf-8') for x in labels]
    return encodings.astype('float32'), labels