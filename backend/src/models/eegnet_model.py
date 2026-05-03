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
        Inisialisasi Parameter Model EEGNet-8,2
        """
        self.nb_classes = nb_classes
        self.channels = channels
        self.samples = samples
        self.dropout_rate = dropout_rate
        
        # [PERBAIKAN] kern_length dinamis (separuh dari samples per second target)
        self.kern_length = kern_length if kern_length else (self.samples // 2)
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        self.model = self.build_model()

    def build_model(self):
        """Membangun Arsitektur Keras dari EEGNet"""
        input_layer = Input(shape=(self.channels, self.samples, 1))

        # ========================================================
        # BLOCK 1: Temporal Convolution & Depthwise Spatial Convolution
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
        
        # Pooling 1 (Tetap 1, 4)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = Dropout(self.dropout_rate)(block1)

        # ========================================================
        # BLOCK 2: Separable Convolution
        # ========================================================
        block2 = SeparableConv2D(self.F2, (1, 16),
                                 use_bias=False, padding='same')(block1)
        block2 = BatchNormalization(axis=1)(block2)
        block2 = Activation('elu')(block2)
        
        # [PERBAIKAN] Pooling 2 Dinamis untuk mencegah Crash pada data E3 (ERP N400)
        # Jika panjang sampel setelah pooling 1 kurang dari 16, gunakan pooling yang lebih kecil
        sisa_sampel = self.samples // 4
        pool2_size = 8 if sisa_sampel >= 16 else (4 if sisa_sampel >= 8 else 2)
        
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
        Melatih model menggunakan data yang telah di-epoch.
        Dilengkapi dengan mekanisme pelindung Plateau & Early Stopping dinamis.
        """
        # [PERBAIKAN] Kurangi Learning Rate jika model mentok sebelum menyerah
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001, verbose=0
        )
        
        # [PERBAIKAN] Early Stopping patience ditingkatkan agar model punya ruang fluktuasi
        # Trik: Jika epochs kecil (saat Optuna tuning), gunakan patience kecil. Jika produksi (500), patience panjang.
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