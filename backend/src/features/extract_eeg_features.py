import numpy as np
from scipy import stats, signal

FEATURE_GROUPS = ['time', 'hjorth', 'barlow', 'band_ratio', 'all']

class EEGFeatureExtractor:
    def __init__(self, fs=256):
        self.fs = fs
        self.bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

    # 1. TIME DOMAIN ANALYSIS
    def _time_domain(self, x):
        return [
            np.mean(x),
            np.var(x),
            stats.skew(x),
            stats.kurtosis(x)
        ]

    # 2. HJORTH PARAMETERS
    def _hjorth(self, x):
        activity = np.var(x)
        dx = np.diff(x)
        ddx = np.diff(dx)
        
        var_dx = np.var(dx)
        mobility = np.sqrt(var_dx / activity) if activity > 0 else 0
        
        var_ddx = np.var(ddx)
        mobility_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else 0
        complexity = mobility_dx / mobility if mobility > 0 else 0
        
        return [activity, mobility, complexity]

    # 3. BARLOW PARAMETERS
    def _barlow(self, x):
        dx = np.diff(x)
        activity = np.var(x)
        mobility = np.var(dx) / activity if activity > 0 else 0
        
        barlow_amp = np.sqrt(activity)
        barlow_freq = np.sqrt(mobility) / (2 * np.pi)
        
        return [barlow_amp, barlow_freq]

    # 4. FREQUENCY RATIO (Band Power Ratio)
    def _band_power_ratio(self, x):
        freqs, psd = signal.welch(x, self.fs, nperseg=min(len(x), 256))
        
        powers = {}
        for band, (low, high) in self.bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            powers[band] = np.sum(psd[idx])
            
        eps = 1e-8
        ratios = [
            powers['alpha'] / (powers['theta'] + eps),
            powers['beta'] / (powers['alpha'] + eps), 
            powers['gamma'] / (powers['beta'] + eps)  
        ]
        return ratios

    # 5. DFA (Dipertahankan murni sebagai referensi akademis, tidak dieksekusi)
    def _dfa(self, x):
        x_cumsum = np.cumsum(x - np.mean(x))
        scales = np.arange(4, min(len(x)//4, 64))
        fluctuations = []
        
        for scale in scales:
            shape = (len(x_cumsum) // scale, scale)
            windows = x_cumsum[:shape[0] * shape[1]].reshape(shape)
            
            x_ax = np.arange(scale)
            rms_list = []
            for w in windows:
                poly = np.polyfit(x_ax, w, 1)
                trend = np.polyval(poly, x_ax)
                rms = np.sqrt(np.mean((w - trend)**2))
                rms_list.append(rms)
            
            fluctuations.append(np.mean(rms_list))
            
        if len(scales) > 1 and len(fluctuations) > 1:
            coeffs = np.polyfit(np.log2(scales), np.log2(fluctuations), 1)
            return [coeffs[0]]
        return [0]

    # 6. PUCK (Dipertahankan murni sebagai referensi akademis, tidak dieksekusi)
    def _puck(self, x):
        x_t = x[:-1]
        x_t1 = x[1:]
        velocity = (x_t1 - x_t) * self.fs
        kinetic = 0.5 * velocity**2
        
        dv = np.diff(velocity)
        pos_accel = np.sum(dv[dv > 0])
        neg_accel = np.abs(np.sum(dv[dv < 0]))
        
        unbalance_ratio = pos_accel / (neg_accel + 1e-8)
        mean_kinetic = np.mean(kinetic)
        puck_score = mean_kinetic * unbalance_ratio
        
        return [mean_kinetic, unbalance_ratio, puck_score]

    def extract_channel_features(self, x, groups=None):
        """Input x: sinyal 1D untuk 1 channel (panjang = Time)."""
        # [PERBAIKAN AUDIT] Hanya memanggil 4 fitur utama jika 'all' dipilih
        if groups is None or 'all' in groups:
            groups = ['time', 'hjorth', 'barlow', 'band_ratio']
            
        all_feats = {
            'time': self._time_domain(x),        
            'hjorth': self._hjorth(x),           
            'barlow': self._barlow(x),           
            'band_ratio': self._band_power_ratio(x), 
            'dfa': self._dfa(x),                
            'puck': self._puck(x)               
        }
        
        features = []
        for g in groups:
            if g in all_feats:
                features.extend(all_feats[g])
        return np.array(features)

    def transform(self, X_3d, groups=None):
        """
        Input: X_3d shape (N, Channels, Time)
        """
        group_names = "Semua Fitur Utama" if (groups is None or 'all' in groups) else f"Fitur {groups}"
        print(f"[*] Mengekstrak {group_names} dari {X_3d.shape[0]} sampel data...")
        
        num_samples, num_channels, _ = X_3d.shape
        X_features = []
        
        for i in range(num_samples):
            sample_features = []
            for ch in range(num_channels):
                signal_ch = X_3d[i, ch, :] 
                ch_feats = self.extract_channel_features(signal_ch, groups)
                sample_features.extend(ch_feats)
            X_features.append(sample_features)
            
        X_features_np = np.array(X_features)
        print(f"[+] Ekstraksi selesai. Dimensi matriks fitur: {X_features_np.shape}")
        return X_features_np