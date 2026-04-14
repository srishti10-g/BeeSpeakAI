import numpy as np
import librosa
import librosa.display
from scipy import signal
import matplotlib.pyplot as plt

class BeeAudioProcessor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path):
        """Load audio file and normalize"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio {file_path}: {e}")
            # Return silent audio as fallback
            return np.zeros(self.sample_rate * 5), self.sample_rate
    
    def preprocess_audio(self, audio):
        """Noise reduction and normalization"""
        try:
            # Apply bandpass filter for bee frequencies (150-500 Hz typical bee buzz)
            nyquist = self.sample_rate / 2
            low = 150 / nyquist
            high = 500 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_audio = signal.filtfilt(b, a, audio)
            
            # Normalize
            normalized_audio = filtered_audio / (np.max(np.abs(filtered_audio)) + 1e-8)
            return normalized_audio
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return audio
    
    def segment_audio(self, audio, segment_length=5):
        """Split audio into segments for analysis"""
        try:
            segment_samples = segment_length * self.sample_rate
            segments = []
            
            for start in range(0, len(audio), segment_samples):
                end = start + segment_samples
                if end <= len(audio):
                    segments.append(audio[start:end])
            
            return segments
        except Exception as e:
            print(f"Error segmenting audio: {e}")
            return [audio]
