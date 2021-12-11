import sys
import numpy as np
import tensorflow as tf
import os
# prevent warning logs about tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from collections import deque
import h5py
import random

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model


class DQNAgent:
    def __init__(self,batch_size):
        self.batch_size=batch_size
        self.learning_rate = 0.0002
        self._model = self._build_model()
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_1 = keras.Input(shape=(20, 10, 1))
        x1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_1)
        x1 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x1)
        x1 = Flatten()(x1)

        input_2 = keras.Input(shape=(20, 10, 1))
        x2 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_2)
        x2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x2)
        x2 = Flatten()(x2)

        x = keras.layers.concatenate([x1, x2])
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(5, activation='linear')(x)

        model = Model(inputs=[input_1, input_2], outputs=x)
        model.compile(optimizer=keras.optimizers.RMSprop(lr=self.learning_rate), loss='mse')

        return model

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        #state = np.reshape(state, [1, 2,16,10,1])
        return self._model.predict(state)
      

    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states)
    
    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)

    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


class TestModel:
    def __init__(self, model_path):
        self._model = self._load_my_model(model_path)

    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')

        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        return self._model.predict(state)

    @property
    def input_dim(self):
        return self._input_dim
