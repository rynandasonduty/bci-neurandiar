import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Flatten, Dropout,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Conv2D, SeparableConv2D, DepthwiseConv2D)
from tensorflow.keras.constraints import max_norm
import numpy as np

class EEGNetClassifier:
    def __init__(self, nb_classes=19, channels=14, samples=256,
                 dropout_rate=0.5, kern_length=None, F1=8, D=2, F2=16):
        """
        EEGNet-8,2 classifier for multichannel EEG decoding.

        Args:
            nb_classes (int): Number of output classes.
            channels (int): Number of EEG channels.
            samples (int): Number of time-domain samples per epoch.
            dropout_rate (float): Dropout probability applied after each pooling block.
            kern_length (int or None): Temporal convolution kernel length. Defaults to samples // 2.
            F1 (int): Number of temporal filters.
            D (int): Depthwise spatial multiplier.
            F2 (int): Number of separable filters.
        """
        self.nb_classes = nb_classes
        self.channels = channels
        self.samples = samples
        self.dropout_rate = dropout_rate

        # Dynamic kern_length: defaults to half the sample length to capture ~2 Hz resolution
        self.kern_length = kern_length if kern_length else (self.samples // 2)

        self.F1 = F1
        self.D = D
        self.F2 = F2

        self.model = self.build_model()

    def build_model(self):
        """Construct the EEGNet Keras model graph."""
        input_layer = Input(shape=(self.channels, self.samples, 1))

        # ========================================================
        # BLOCK 1: Temporal Convolution + Depthwise Spatial Convolution
        # ========================================================
        block1 = Conv2D(self.F1, (1, self.kern_length), padding='same',
                        input_shape=(self.channels, self.samples, 1),
                        use_bias=False)(input_layer)
        block1 = BatchNormalization(axis=1)(block1)

        block1 = DepthwiseConv2D((self.channels, 1), use_bias=False,
                                 depth_multiplier=self.D,
                                 depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization(axis=1)(block1)
        block1 = Activation('elu')(block1)

        block1 = AveragePooling2D((1, 4))(block1)
        block1 = Dropout(self.dropout_rate)(block1)

        # ========================================================
        # BLOCK 2: Separable Convolution
        # ========================================================
        block2 = SeparableConv2D(self.F2, (1, 16),
                                 use_bias=False, padding='same')(block1)
        block2 = BatchNormalization(axis=1)(block2)
        block2 = Activation('elu')(block2)

        # Dynamic pooling size to prevent a shape crash for short ERP windows (E3)
        remaining_samples = self.samples // 4
        pool2_size = 8 if remaining_samples >= 16 else (4 if remaining_samples >= 8 else 2)

        block2 = AveragePooling2D((1, pool2_size))(block2)
        block2 = Dropout(self.dropout_rate)(block2)

        # ========================================================
        # CLASSIFICATION BLOCK
        # ========================================================
        flatten = Flatten(name='flatten')(block2)

        dense = Dense(self.nb_classes, name='dense',
                      kernel_constraint=max_norm(0.25))(flatten)
        softmax_output = Activation('softmax', name='softmax')(dense)

        model = Model(inputs=input_layer, outputs=softmax_output)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=500, batch_size=32):
        """
        Train the EEGNet model with adaptive learning rate and early stopping.

        Early stopping patience scales with epoch budget: 50 epochs for production
        runs (>=200 epochs) or 30% of the epoch budget for Optuna trial runs.
        """
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001, verbose=0
        )

        es_patience = 50 if epochs >= 200 else int(epochs * 0.3)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=es_patience, restore_best_weights=True, verbose=1
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[reduce_lr, early_stopping],
            verbose=1
        )
        return history

    def predict_probabilities(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, filepath="eegnet_model.h5"):
        self.model.save(filepath)
