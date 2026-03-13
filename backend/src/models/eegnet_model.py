import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Flatten, Dropout, 
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Conv2D, SeparableConv2D, DepthwiseConv2D)
from tensorflow.keras.constraints import max_norm
import numpy as np

class EEGNetClassifier:
    def __init__(self, nb_classes=19, channels=14, samples=256, 
                 dropout_rate=0.5, kern_length=128, F1=8, D=2, F2=16):
        """
        Inisialisasi Parameter Model EEGNet-8,2
        - nb_classes: 19 Suku Kata Target
        - channels: 14 Sensor Emotiv EPOC X
        - samples: 256 (Jendela waktu 1.0 detik)
        - dropout_rate: 0.5 untuk mencegah overfitting (Regularisasi)
        - kern_length: 128 (Setengah dari sampling rate 256Hz)
        - F1: 8 (Jumlah filter temporal)
        - D: 2 (Kedalaman filter spasial per filter temporal)
        - F2: 16 (Jumlah pointwise filter, idealnya F1 * D)
        """
        self.nb_classes = nb_classes
        self.channels = channels
        self.samples = samples
        self.dropout_rate = dropout_rate
        self.kern_length = kern_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        self.model = self.build_model()

    def build_model(self):
        """Membangun Arsitektur Keras dari EEGNet"""
        # Input Layer: Tensor 2D (Channels x Time Samples)
        input_layer = Input(shape=(self.channels, self.samples, 1))

        # ========================================================
        # BLOCK 1: Temporal Convolution & Depthwise Spatial Convolution
        # ========================================================
        
        # 1. Temporal Convolution (Berfungsi sebagai filter bank frekuensi)
        block1 = Conv2D(self.F1, (1, self.kern_length), padding='same',
                        input_shape=(self.channels, self.samples, 1),
                        use_bias=False)(input_layer)
        block1 = BatchNormalization(axis=1)(block1)
        
        # 2. Depthwise Spatial Convolution (Mempelajari kombinasi spasial sensor)
        # max_norm(1) digunakan untuk menstabilkan pembobotan spasial
        block1 = DepthwiseConv2D((self.channels, 1), use_bias=False, 
                                 depth_multiplier=self.D,
                                 depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization(axis=1)(block1)
        block1 = Activation('elu')(block1)
        
        # 3. Pooling & Dropout (Pengurangan dimensi 1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = Dropout(self.dropout_rate)(block1)

        # ========================================================
        # BLOCK 2: Separable Convolution
        # ========================================================
        
        # 4. Separable Convolution (Mengurangi dimensi dan mencampur peta fitur)
        block2 = SeparableConv2D(self.F2, (1, 16),
                                 use_bias=False, padding='same')(block1)
        block2 = BatchNormalization(axis=1)(block2)
        block2 = Activation('elu')(block2)
        
        # 5. Pooling & Dropout (Pengurangan dimensi 2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = Dropout(self.dropout_rate)(block2)

        # ========================================================
        # CLASSIFICATION BLOCK
        # ========================================================
        
        # 6. Meratakan tensor menjadi array 1D
        flatten = Flatten(name='flatten')(block2)
        
        # 7. Dense Layer (Lapisan Output dengan fungsi Softmax)
        # max_norm(0.25) digunakan untuk regularisasi klasifikasi akhir
        dense = Dense(self.nb_classes, name='dense', 
                      kernel_constraint=max_norm(0.25))(flatten)
        softmax_output = Activation('softmax', name='softmax')(dense)

        # Merakit dan Mengkompilasi Model
        model = Model(inputs=input_layer, outputs=softmax_output)
        
        # Optimizer Adam dengan categorical crossentropy untuk klasifikasi multi-kelas
        model.compile(loss='sparse_categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=500, batch_size=32):
        """
        Melatih model menggunakan data yang telah di-epoch.
        X_train shape harus: (samples, 14, 256, 1)
        """
        # Menambahkan callback Early Stopping untuk mencegah overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=30, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        return history
    
    def predict_probabilities(self, X_test):
        """
        Mengembalikan vektor probabilitas (Softmax) untuk digunakan 
        oleh Regresi Logistik di tahap perakitan kata (Word Assembler).
        """
        return self.model.predict(X_test)

    def save_model(self, filepath="eegnet_model.h5"):
        self.model.save(filepath)

if __name__ == "__main__":
    # Skenario Simulasi/Dry-Run Struktur Matriks
    print("="*50)
    print(" MENGINISIALISASI ARSITEKTUR EEGNET ")
    print("="*50)
    
    # Membangun objek AI
    eegnet = EEGNetClassifier()
    
    # Menampilkan ringkasan layer dan kalkulasi parameter
    eegnet.model.summary()
    
    print("\n[*] Membuat data simulasi (Dummy Tensor)...")
    # Mensimulasikan 10 epoch data EEG bersih: (10 sampel, 14 sensor, 256 milidetik, 1 channel depth)
    dummy_X = np.random.randn(10, 14, 256, 1)
    
    print("[*] Melakukan prediksi uji coba...")
    # Mendapatkan vektor probabilitas untuk Slot 1 dan Slot 2
    dummy_P1_P2 = eegnet.predict_probabilities(dummy_X)
    
    print(f"\n[+] Bentuk Input Tensor   : {dummy_X.shape}")
    print(f"[+] Bentuk Output Softmax : {dummy_P1_P2.shape}")
    print(f"[+] Output Vektor (Baris 1) menunjukkan distribusi probabilitas untuk 19 suku kata.")
    print("="*50)
    print(" ARSITEKTUR SIAP DIGUNAKAN! ")