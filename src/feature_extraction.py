import numpy as np
import librosa
from scipy import stats

class BeeFeatureExtractor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.mfcc_params = {
            'n_mfcc': 13,
            'n_fft': 2048,
            'hop_length': 512
        }
    
    def extract_all_features(self, audio):
        """Extract comprehensive feature set from bee audio"""
        features = {}
        
        try:
            # Spectral features
            features.update(self._spectral_features(audio))
            
            # Temporal features
            features.update(self._temporal_features(audio))
            
            # MFCC features
            features.update(self._mfcc_features(audio))
            
            # Special bee-specific features
            features.update(self._bee_specific_features(audio))
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return default features
            features = self._get_default_features()
        
        return features
    
    def _spectral_features(self, audio):
        """Extract spectral domain features"""
        try:
            stft = np.abs(librosa.stft(audio))
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_flatness_mean': float(np.mean(librosa.feature.spectral_flatness(y=audio))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio)))
            }
        except:
            return self._get_default_spectral_features()
    
    def _temporal_features(self, audio):
        """Extract temporal domain features"""
        try:
            rms = librosa.feature.rms(y=audio)
            return {
                'rms_energy': float(np.mean(rms)),
                'rms_energy_std': float(np.std(rms)),
                'amplitude_envelope_mean': float(np.mean(np.abs(audio))),
                'amplitude_envelope_std': float(np.std(np.abs(audio)))
            }
        except:
            return self._get_default_temporal_features()
    
    def _mfcc_features(self, audio):
        """Extract MFCC features"""
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, **self.mfcc_params)
            mfcc_features = {}
            
            for i in range(min(5, mfccs.shape[0])):  # Only first 5 MFCCs
                mfcc_features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                mfcc_features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            
            return mfcc_features
        except:
            return self._get_default_mfcc_features()
    
    def _bee_specific_features(self, audio):
        """Bee-specific acoustic features"""
        try:
            # Fundamental frequency estimation (pitch of bee buzz)
            f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                        fmin=150, 
                                                        fmax=500, 
                                                        sr=self.sample_rate,
                                                        frame_length=2048)
            f0_valid = f0[~np.isnan(f0)]
            
            # Harmonic-to-noise ratio (indication of stress/health)
            harmonic, percussive = librosa.effects.hpss(audio)
            hnr = np.mean(harmonic) / (np.mean(percussive) + 1e-8)
            
            return {
                'fundamental_freq_mean': float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0,
                'fundamental_freq_std': float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0,
                'harmonic_noise_ratio': float(hnr),
                'activity_ratio': float(len(f0_valid) / len(audio) * self.sample_rate) if len(f0_valid) > 0 else 0.0
            }
        except:
            return self._get_default_bee_features()
    
    def _get_default_features(self):
        """Return default feature set when extraction fails"""
        features = {}
        features.update(self._get_default_spectral_features())
        features.update(self._get_default_temporal_features())
        features.update(self._get_default_mfcc_features())
        features.update(self._get_default_bee_features())
        return features
    
    def _get_default_spectral_features(self):
        return {
            'spectral_centroid_mean': 250.0,
            'spectral_centroid_std': 50.0,
            'spectral_rolloff_mean': 4000.0,
            'spectral_bandwidth_mean': 1000.0,
            'spectral_flatness_mean': 0.5,
            'zero_crossing_rate': 0.1
        }
    
    def _get_default_temporal_features(self):
        return {
            'rms_energy': 0.1,
            'rms_energy_std': 0.02,
            'amplitude_envelope_mean': 0.05,
            'amplitude_envelope_std': 0.01
        }
    
    def _get_default_mfcc_features(self):
        features = {}
        for i in range(1, 6):
            features[f'mfcc_{i}_mean'] = 0.0
            features[f'mfcc_{i}_std'] = 1.0
        return features
    
    def _get_default_bee_features(self):
        return {
            'fundamental_freq_mean': 220.0,
            'fundamental_freq_std': 0.2,
            'harmonic_noise_ratio': 1.5,
            'activity_ratio': 0.5
        }
