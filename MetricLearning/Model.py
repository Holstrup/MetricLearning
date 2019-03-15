import keras
import keras.backend as K


class Model:
    def __init__(self):
        self.model = self.build_model()
        self.compile_model()

    def build_model(self, embedding_size=128):
        base_model = keras.applications.ResNet50(weights='imagenet', include_top=False,
                                                 input_tensor=keras.layers.Input(shape=(224, 224, 3)))
        x = base_model.output
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(embedding_size, activation='softmax')(x)
        full_model = keras.Model(inputs=base_model.input, outputs=x)
        return full_model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss=keras.losses.MSE,
                           metrics=['accuracy'])


def chi_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

