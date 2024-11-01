import os

import numpy as np
import tensorflow as tf

from contextlib import redirect_stdout
from tensorflow.keras import backend as K
from tensorflow.keras import Model

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Activation, MaxPooling2D, Reshape

class HTRModel:

    def __init__(self,
                 architecture,
                 input_size,
                 vocab_size,
                 greedy=False,
                 beam_width=10,
                 top_paths=1,
                 stop_tolerance=20,
                 reduce_tolerance=15,
                 cooldown=0):

        self.architecture = globals()[architecture]
        self.input_size = input_size
        self.vocab_size = vocab_size #97

        self.model = None
        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = max(1, top_paths)

        self.stop_tolerance = stop_tolerance
        self.reduce_tolerance = reduce_tolerance
        self.cooldown = cooldown

    def summary(self, output=None, target=None):

        self.model.summary()

        if target is not None:
            os.makedirs(output, exist_ok=True)

            with open(os.path.join(output, target), "w") as f:
                with redirect_stdout(f):
                    self.model.summary()

    def load_checkpoint(self, target):

        if os.path.isfile(target):
            if self.model is None:
                self.compile()

            self.model.load_weights(target)

    def get_callbacks(self, logdir, checkpoint, monitor="val_loss", verbose=0):

        callbacks = [
            CSVLogger(
                filename=os.path.join(logdir, "epochs.log"),
                separator=";",
                append=True),
            ModelCheckpoint(
                filepath=checkpoint,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose),
            EarlyStopping(
                monitor=monitor,
                min_delta=1e-8,
                patience=self.stop_tolerance,
                restore_best_weights=True,
                verbose=verbose),
            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=1e-8,
                factor=0.2,
                patience=self.reduce_tolerance,
                cooldown=self.cooldown,
                verbose=verbose)
        ]

        return callbacks

    def compile(self, learning_rate=None, initial_step=0, target=None, output=None):
        inputs, outputs = self.architecture(self.input_size, self.vocab_size + 1)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer, loss=self.ctc_loss_lambda_func)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs):

        out = self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                             callbacks=callbacks, validation_split=validation_split,
                             validation_data=validation_data, shuffle=shuffle,
                             class_weight=class_weight, sample_weight=sample_weight,
                             initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                             validation_steps=validation_steps, validation_freq=validation_freq,
                             max_queue_size=max_queue_size, workers=workers,
                             use_multiprocessing=use_multiprocessing, **kwargs)
        return out

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=1,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                ctc_decode=True):

        if verbose == 1:
            print("Model Predict")
        out = self.model.predict(x=x, batch_size=batch_size, verbose=verbose, steps=steps,
                                 callbacks=callbacks, max_queue_size=max_queue_size,
                                 workers=workers, use_multiprocessing=use_multiprocessing)
       
        steps_done = 0
        if verbose == 1:
            print("CTC Decode")
            progbar = tf.keras.utils.Progbar(target=steps)

        batch_size = int(np.ceil(len(out) / steps))
        input_length = len(max(out, key=len))

        predicts, probabilities = [], []
        while steps_done < steps:
            index = steps_done * batch_size
            until = index + batch_size

            x_test = np.asarray(out[index:until])
            x_test_len = np.asarray([input_length for _ in range(len(x_test))])

            decode, log = K.ctc_decode(x_test,
                                       x_test_len,
                                       greedy=self.greedy,
                                       beam_width=self.beam_width,
                                       top_paths=self.top_paths)

            probabilities.extend([np.exp(x) for x in log])
            decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            predicts.extend(np.swapaxes(decode, 0, 1))

            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

        return (predicts, probabilities)

    @staticmethod
    def ctc_loss_lambda_func(y_true, y_pred):
        """Function for computing the CTC loss"""

        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)

        # y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
        # output of every model is softmax
        # so sum across alphabet_size_1_hot_encoded give 1
        #               string_length give string length
        input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
        input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)

        # y_true strings are padded with 0
        # so sum of non-zero gives number of characters in this string
        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

        # average loss across all entries in the batch
        loss = tf.reduce_mean(loss)

        return loss

def cnn_bilstm(input_size, d_model):
    input_data = Input(name="input", shape=input_size)
    
    cnn = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding="same")(input_data)
    cnn = BatchNormalization(axis = -1)(cnn)

    cnn = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Dropout(rate=0.1)(cnn)

    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2))(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Dropout(rate=0.1)(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Dropout(rate=0.1)(cnn)

    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2))(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Dropout(rate=0.1)(cnn)
    
    cnn = MaxPooling2D(pool_size=(2,1), strides=(2,1))(cnn)
    
    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)
    
    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5))(blstm)
    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5))(blstm)
    
    output_data = Dense(units=d_model, activation="softmax")(blstm)
    
    return (input_data, output_data)

def fajardo(input_size, d_model):
    input_data = Input(name="input", shape=input_size)
    
    cnn = Conv2D(filters=16, kernel_size=(3,3), padding="same")(input_data)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = Conv2D(filters=16, kernel_size=(3,3), padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), padding="same")(cnn)    
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2))(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3,3), padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3,3), padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3,3), padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = MaxPooling2D(pool_size=(2,2), strides=(2,2))(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3,3), padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3,3), padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3,3), padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = MaxPooling2D(pool_size=(2,1), strides=(2,1))(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3,3), padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = Conv2D(filters=256, kernel_size=(3,3), padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = Conv2D(filters=256, kernel_size=(3,3), padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = Conv2D(filters=256, kernel_size=(3,3), padding="same")(cnn)
    cnn = BatchNormalization(axis = -1)(cnn)
    cnn = Activation("relu")(cnn)

    cnn = MaxPooling2D(pool_size=(2,1), strides=(2,1))(cnn)
    
    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)
    
    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25))(blstm)
    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25))(blstm)
    blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25))(blstm)

    output_data = Dense(units=d_model, activation="softmax")(blstm)

    return (input_data, output_data)

