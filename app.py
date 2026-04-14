# app.py (COMPLETE STORAGE INTEGRATION SOLUTION)
import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import base64
from io import BytesIO
import datetime
import json
from PIL import Image
import time
from supabase import create_client, Client
import math
from decimal import Decimal
import requests
from urllib.parse import quote

# Supabase configuration
SUPABASE_URL = "https://twnvqbuhnitxxuoxmjwj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR3bnZxYnVobml0eHh1b3htandqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzY2NTQ0OSwiZXhwIjoyMDc5MjQxNDQ5fQ.i2N6jYPRaW7XLUpyVogzJX3tNfNe0NxYb7BPQoHnXSg"

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Supabase initialization failed: {e}")
        return None

supabase = init_supabase()

# Import our custom modules
try:
    from src.audio_processing import BeeAudioProcessor
    from src.feature_extraction import BeeFeatureExtractor
    from src.clustering_model import BeeClusterAnalyzer
    from src.visualization import BeeHealthVisualizer
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.info("Please make sure all the required source files are in the 'src' directory.")
    
    # Create fallback classes if imports fail
    class BeeAudioProcessor:
        def __init__(self, sample_rate=22050):
            self.sample_rate = sample_rate
        def load_audio(self, file_path):
            return np.zeros(self.sample_rate * 5), self.sample_rate
        def preprocess_audio(self, audio):
            return audio
    
    class BeeFeatureExtractor:
        def extract_all_features(self, audio):
            return {'error': 'Feature extraction failed'}
    
    class BeeClusterAnalyzer:
        def prepare_features(self, feature_data):
            return np.eye(len(feature_data))
        def cluster_analysis(self, features, n_clusters=3):
            return np.random.randint(0, n_clusters, len(features))
        def interpret_clusters(self, features, cluster_labels):
            return {0: {'inferred_status': 'Analysis Failed'}}
        def reduce_dimensionality(self, features, method='pca'):
            return np.random.rand(len(features), 2)
    
    class BeeHealthVisualizer:
        def plot_cluster_analysis(self, reduced_features, cluster_labels, cluster_profiles):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Visualization module not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig

# -------------------------
# STORAGE INTEGRATION FUNCTIONS
# -------------------------

def download_audio_from_storage(storage_path):
    """Download audio file from Supabase Storage"""
    try:
        # Construct the public URL for the file
        encoded_path = quote(storage_path)
        storage_url = f"{SUPABASE_URL}/storage/v1/object/public/audio-files/{encoded_path}"
        
        # Download the file
        response = requests.get(storage_url)
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Failed to download file from storage: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error downloading from storage: {e}")
        return None

def get_storage_files():
    """Get list of all files in the audio-files bucket"""
    try:
        # Use Supabase storage API to list files
        response = supabase.storage.from_('audio-files').list()
        return response
    except Exception as e:
        st.error(f"Error listing storage files: {e}")
        return []

def sync_storage_to_database():
    """Sync files from storage to database table"""
    try:
        storage_files = get_storage_files()
        if not storage_files:
            st.warning("No files found in storage bucket 'audio-files'")
            return 0
        
        synced_count = 0
        for file_info in storage_files:
            # Check if file already exists in database
            existing = supabase.table("audio_recordings")\
                .select("id")\
                .eq("storage_path", file_info['name'])\
                .execute()
            
            if not existing.data:
                # Insert new record
                insert_data = {
                    "device_id": "550e8400-e29b-41d4-a716-446655440000",
                    "hive_id": "550e8400-e29b-41d4-a716-446655440001",
                    "recording_time": datetime.datetime.now().isoformat(),
                    "duration_seconds": 30,
                    "sample_rate": 22050,
                    "storage_path": file_info['name'],
                    "file_name": file_info['name'],
                    "file_size_bytes": file_info.get('metadata', {}).get('size', 0)
                }
                
                result = supabase.table("audio_recordings").insert(insert_data).execute()
                if result.data:
                    synced_count += 1
        
        return synced_count
    except Exception as e:
        st.error(f"Error syncing storage to database: {e}")
        return 0

# -------------------------
# JSON-safety helpers
# -------------------------
def sanitize_for_json(obj, replace_with=None):
    """Recursively convert numpy/decimal types to native Python types"""
    if obj is None:
        return None

    try:
        import numpy as _np
        if isinstance(obj, (_np.floating, _np.integer, _np.bool_)):
            try:
                obj = obj.item()
            except Exception:
                try:
                    obj = float(obj)
                except Exception:
                    return replace_with
    except Exception:
        pass

    if isinstance(obj, Decimal):
        try:
            obj = float(obj)
        except Exception:
            return replace_with

    if isinstance(obj, float):
        if not math.isfinite(obj):
            return replace_with
        return float(obj)

    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, bool):
        return bool(obj)

    if isinstance(obj, str):
        return obj

    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v, replace_with=replace_with) for v in obj]

    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            key = str(k)
            sanitized[key] = sanitize_for_json(v, replace_with=replace_with)
        return sanitized

    try:
        if hasattr(obj, "__float__"):
            val = float(obj)
            if math.isfinite(val):
                return val
            return replace_with
    except Exception:
        pass

    try:
        if hasattr(obj, "__int__"):
            return int(obj)
    except Exception:
        pass

    return str(obj)

def find_non_finite(obj, path="root"):
    """Debug helper: return list of (path, value) where value is NaN or Inf."""
    issues = []
    if obj is None:
        return issues

    if isinstance(obj, dict):
        for k, v in obj.items():
            issues += find_non_finite(v, path + f".{k}")
        return issues

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            issues += find_non_finite(v, path + f"[{i}]")
        return issues

    try:
        import numpy as _np
        if isinstance(obj, (_np.floating, _np.integer, _np.bool_)):
            try:
                obj = obj.item()
            except Exception:
                pass
    except Exception:
        pass

    if isinstance(obj, float):
        if not math.isfinite(obj):
            issues.append((path, obj))
    return issues

# Enhanced CSS with modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B00;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #FF6B00, #FF8C00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .real-time-alert {
        background: linear-gradient(45deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        animation: pulse 2s infinite;
        border-left: 5px solid #c44569;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .status-healthy {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #2E7D32;
    }
    .status-warning {
        background: linear-gradient(45deg, #FF9800, #F57C00);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #EF6C00;
    }
    .status-danger {
        background: linear-gradient(45deg, #F44336, #D32F2F);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #C62828;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        background-color: #f8fff8;
        transition: all 0.3s ease;
    }
    .upload-box:hover {
        border-color: #FF6B00;
        background-color: #fff8f0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #FF6B00;
        margin: 10px 0;
    }
    .nav-item {
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-item:hover {
        background-color: #fff0e0;
    }
    .nav-item.active {
        background-color: #FF6B00;
        color: white;
    }
    .history-item {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    .explanation-box {
        background: #808080;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #FF6B00;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_authenticated' not in st.session_state:
    st.session_state.user_authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Real-Time Dashboard"
if 'last_checked' not in st.session_state:
    st.session_state.last_checked = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# Supabase helper functions
def get_recent_recordings(limit=10):
    """Get recent recordings from Supabase"""
    try:
        if supabase is None:
            return []
        response = supabase.table("audio_recordings")\
            .select("*, devices(name), hives(name)")\
            .order("recording_time", desc=True)\
            .limit(limit)\
            .execute()
        return response.data
    except Exception as e:
        st.error(f"Error fetching recordings: {e}")
        return []

def get_analysis_results(limit=20):
    """Get recent analysis results"""
    try:
        if supabase is None:
            return []
        response = supabase.table("analysis_results")\
            .select("*, audio_recordings(recording_time, devices(name)), hives(name)")\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        return response.data
    except Exception as e:
        st.error(f"Error fetching analysis results: {e}")
        return []

def save_analysis_result(recording_id, analysis_data):
    """Save analysis results to Supabase"""
    try:
        if supabase is None:
            st.warning("Supabase client not initialized. Skipping save.")
            return None

        result = {
            "recording_id": recording_id,
            "hive_id": "550e8400-e29b-41d4-a716-446655440001",
            "health_status": analysis_data.get('status', 'Unknown'),
            "confidence_score": analysis_data.get('confidence', 0.0),
            "fundamental_freq_mean": analysis_data.get('metrics', {}).get('frequency', 0.0),
            "harmonic_noise_ratio": analysis_data.get('metrics', {}).get('harmony', 0.0),
            "activity_ratio": analysis_data.get('metrics', {}).get('activity', 0.0),
            "cluster_group": int(analysis_data.get('cluster', 0)) if analysis_data.get('cluster') is not None else None,
            "analysis_details": analysis_data
        }

        issues = find_non_finite(result)
        if issues:
            st.warning(f"Found non-finite values in result before sanitize: {issues}")

        sanitized_result = sanitize_for_json(result, replace_with=None)

        cs = sanitized_result.get("confidence_score")
        if isinstance(cs, (int, float)):
            if not math.isfinite(cs):
                sanitized_result["confidence_score"] = None
            else:
                sanitized_result["confidence_score"] = max(0.0, min(1.0, float(cs)))

        response = supabase.table("analysis_results").insert(sanitized_result).execute()
        return response.data
    except Exception as e:
        st.error(f"Error saving analysis: {e}")
        return None

# -------------------------
# UPDATED ANALYZE FUNCTION WITH STORAGE SUPPORT
# -------------------------

def generate_synthetic_bee_sounds(num_samples=100):
    """Generate synthetic bee feature dicts for demo / clustering"""
    all_features = []
    
    conditions = {
        'healthy': {'freq_mean': 220, 'freq_std': 0.2, 'hnr': 2.5, 'activity': 0.6},
        'stressed': {'freq_mean': 280, 'freq_std': 0.5, 'hnr': 1.2, 'activity': 0.9},
        'queenless': {'freq_mean': 200, 'freq_std': 0.8, 'hnr': 0.8, 'activity': 0.3}
    }
    
    samples_per_condition = max(1, int(num_samples // len(conditions)))
    
    for condition, params in conditions.items():
        for i in range(samples_per_condition):
            f_mean = float(np.random.normal(params['freq_mean'], 10))
            f_std  = float(abs(np.random.normal(params['freq_std'], 0.1)))
            hnr    = float(np.random.normal(params['hnr'], 0.3))
            activity = float(np.random.normal(params['activity'], 0.1))
            
            feat = {
                'fundamental_freq_mean': f_mean,
                'fundamental_freq_std': f_std,
                'harmonic_noise_ratio': hnr,
                'activity_ratio': activity,
                'spectral_centroid_mean': float(np.random.normal(250, 50)),
                'spectral_centroid_std': float(abs(np.random.normal(50, 10))),
                'rms_energy': float(abs(np.random.normal(0.1, 0.02))),
                'rms_energy_std': float(abs(np.random.normal(0.02, 0.005))),
                'zero_crossing_rate': float(abs(np.random.normal(0.1, 0.02)))
            }
            
            for j in range(1, 6):
                feat[f'mfcc_{j}_mean'] = float(np.random.normal(0, 1))
                feat[f'mfcc_{j}_std']  = float(abs(np.random.normal(0.5, 0.1)))
            
            all_features.append(feat)
    
    while len(all_features) < num_samples:
        extra = all_features[0].copy()
        extra['fundamental_freq_mean'] = float(extra['fundamental_freq_mean'] + np.random.normal(0,3))
        all_features.append(extra)
    
    return all_features

def clean_feature_data(feature_data):
    """Ensure feature dicts contain only finite python numerics or None"""
    cleaned_data = []
    for features in feature_data:
        cleaned_features = {}
        if not isinstance(features, dict):
            cleaned_data.append({})
            continue
        for key, value in features.items():
            if isinstance(value, (np.floating, np.integer, np.bool_)):
                try:
                    value = value.item()
                except Exception:
                    try:
                        value = float(value)
                    except Exception:
                        value = None
            if isinstance(value, float):
                if not math.isfinite(value):
                    if 'freq' in key:
                        cleaned_features[key] = 220.0
                    elif 'ratio' in key or 'rate' in key:
                        cleaned_features[key] = 0.5
                    elif 'energy' in key:
                        cleaned_features[key] = 0.1
                    elif 'std' in key:
                        cleaned_features[key] = 1.0
                    elif 'mean' in key:
                        cleaned_features[key] = 0.0
                    else:
                        cleaned_features[key] = 0.0
                else:
                    cleaned_features[key] = float(value)
            elif isinstance(value, int):
                cleaned_features[key] = int(value)
            elif value is None:
                cleaned_features[key] = None
            else:
                try:
                    if isinstance(value, str):
                        cleaned_features[key] = value
                    else:
                        cleaned_features[key] = sanitize_for_json(value, replace_with=None)
                except Exception:
                    cleaned_features[key] = None
        cleaned_data.append(cleaned_features)
    return cleaned_data

def analyze_supabase_recording(recording):
    """Analyze a recording from Supabase - UPDATED FOR STORAGE SUPPORT"""
    try:
        audio_bytes = None
        
        # Check storage path first (new method)
        storage_path = recording.get('storage_path')
        audio_base64 = recording.get('audio_data_base64')
        
        if storage_path:
            # Download from Supabase Storage
            audio_bytes = download_audio_from_storage(storage_path)
            if not audio_bytes:
                st.error(f"Failed to download audio from storage: {storage_path}")
                return None
        elif audio_base64 and audio_base64 != "":
            # Use existing base64 data (backward compatibility)
            try:
                audio_bytes = base64.b64decode(audio_base64)
            except Exception as e:
                st.error(f"Error decoding base64 audio: {e}")
                return None
        else:
            st.error("No audio data available for analysis")
            return None

        # Save to temporary file and process
        tmp_path = None
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

        # Analyze using existing pipeline
        processor = BeeAudioProcessor()
        extractor = BeeFeatureExtractor()

        if tmp_path:
            audio, sr = processor.load_audio(tmp_path)
            processed_audio = processor.preprocess_audio(audio)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        else:
            dummy_len = processor.sample_rate * 5 if hasattr(processor, 'sample_rate') else 22050 * 5
            audio = np.zeros(int(dummy_len), dtype=float)
            processed_audio = processor.preprocess_audio(audio)

        features = extractor.extract_all_features(processed_audio)
        features = sanitize_for_json(features, replace_with=None)

        # Combine with synthetic data for clustering
        synthetic_data = generate_synthetic_bee_sounds(50)
        synthetic_data = [sanitize_for_json(x, replace_with=None) for x in synthetic_data]

        all_features = [features] + synthetic_data
        cleaned_features = clean_feature_data(all_features)

        # Perform clustering analysis
        features_df = pd.DataFrame(cleaned_features)
        clustering_df = features_df.select_dtypes(include=[np.number]).fillna(0)

        analyzer = BeeClusterAnalyzer()
        features_scaled = analyzer.prepare_features(clustering_df)

        try:
            if isinstance(features_scaled, np.ndarray):
                if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
                    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            try:
                features_scaled = np.nan_to_num(np.array(features_scaled, dtype=float), nan=0.0)
            except Exception:
                pass

        n_rows = len(features_scaled) if hasattr(features_scaled, '__len__') else 2
        optimal_clusters = max(2, min(3, n_rows - 1)) if n_rows > 2 else 2

        cluster_labels = analyzer.cluster_analysis(features_scaled, n_clusters=optimal_clusters)
        cluster_profiles = analyzer.interpret_clusters(features_scaled, cluster_labels)

        # First sample is our real recording
        real_cluster = int(cluster_labels[0]) if len(cluster_labels) > 0 else 0
        status = cluster_profiles.get(real_cluster, {}).get('inferred_status', 'Unknown')

        result = {
            'status': status,
            'cluster': real_cluster,
            'metrics': {
                'frequency': cluster_profiles.get(real_cluster, {}).get('fundamental_freq_mean', {}).get('mean', 0),
                'harmony': cluster_profiles.get(real_cluster, {}).get('harmonic_noise_ratio', {}).get('mean', 0),
                'activity': cluster_profiles.get(real_cluster, {}).get('activity_ratio', {}).get('mean', 0)
            },
            'cluster_profiles': cluster_profiles
        }

        result = sanitize_for_json(result, replace_with=None)
        return result

    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None

def analyze_uploaded_audio(uploaded_files):
    """Analyze uploaded audio files"""
    processor = BeeAudioProcessor()
    extractor = BeeFeatureExtractor()

    all_features = []
    file_names = []

    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            audio, sr = processor.load_audio(tmp_path)
            processed_audio = processor.preprocess_audio(audio)
            features = extractor.extract_all_features(processed_audio)
            features['audio_file'] = uploaded_file.name

            features = sanitize_for_json(features, replace_with=None)

            all_features.append(features)
            file_names.append(uploaded_file.name)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue

    return all_features, file_names

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def get_status_color(status):
    """Return color based on health status"""
    if 'healthy' in status.lower():
        return '🟢', 'status-healthy'
    elif 'warning' in status.lower() or 'risk' in status.lower():
        return '🟡', 'status-warning'
    elif 'stress' in status.lower() or 'queenless' in status.lower() or 'parasite' in status.lower():
        return '🔴', 'status-danger'
    else:
        return '⚪', ''

def add_to_history(analysis_data):
    """Add analysis to history"""
    history_item = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'files': analysis_data.get('files', []),
        'status': analysis_data.get('status', 'Unknown'),
        'cluster': analysis_data.get('cluster', 0),
        'metrics': analysis_data.get('metrics', {}),
        'summary': analysis_data.get('summary', '')
    }
    st.session_state.analysis_history.insert(0, history_item)
    
    if len(st.session_state.analysis_history) > 50:
        st.session_state.analysis_history = st.session_state.analysis_history[:50]

def interpret_graph_for_beekeeper(cluster_profiles, real_cluster=None):
    """Convert complex graph data into simple text for beekeepers"""
    interpretations = []
    
    if real_cluster is not None:
        profile = cluster_profiles.get(real_cluster, {})
        status = profile.get('inferred_status', 'Unknown')
        
        interpretations.append(f"### 🎯 Your Hive Analysis Summary")
        interpretations.append(f"**Overall Status:** {status}")
        
        if 'healthy' in status.lower():
            interpretations.append("**What this means:** Your bees are happy and productive! The sound pattern shows normal activity levels with consistent buzzing.")
            interpretations.append("**Key indicators:**")
            interpretations.append("- ✅ Steady, harmonious buzzing sound")
            interpretations.append("- ✅ Consistent activity pattern")
            interpretations.append("- ✅ Normal frequency range (200-250 Hz)")
        elif 'queenless' in status.lower():
            interpretations.append("**What this means:** The hive may be without a queen. Bees sound disorganized and irregular.")
            interpretations.append("**Key indicators:**")
            interpretations.append("- ❌ Irregular buzzing patterns")
            interpretations.append("- ❌ Low harmony in sounds")
            interpretations.append("- ❌ Unusual activity fluctuations")
        elif 'stress' in status.lower():
            interpretations.append("**What this means:** Bees are showing signs of stress, possibly from heat, cold, or disturbances.")
            interpretations.append("**Key indicators:**")
            interpretations.append("- ⚠️ Higher than normal buzzing frequency")
            interpretations.append("- ⚠️ Agitated sound patterns")
            interpretations.append("- ⚠️ Increased activity levels")
        elif 'swarming' in status.lower():
            interpretations.append("**What this means:** The hive may be preparing to swarm. Bees are very active and synchronized.")
            interpretations.append("**Key indicators:**")
            interpretations.append("- 🐝 Very high activity levels")
            interpretations.append("- 🐝 Synchronized buzzing patterns")
            interpretations.append("- 🐝 Specific frequency signatures")
    
    interpretations.append("### 📊 Pattern Groups Found")
    for cluster_id, profile in cluster_profiles.items():
        status = profile.get('inferred_status', 'Unknown')
        interpretations.append(f"**Group {cluster_id + 1}:** {status}")
    
    return "\n\n".join(interpretations)

# -------------------------
# UPDATED REAL-TIME DASHBOARD WITH SYNC BUTTON
# -------------------------

def login_section():
    """User login section"""
    st.sidebar.markdown("### 🔐 User Account")
    
    if not st.session_state.user_authenticated:
        username = st.sidebar.text_input("Username", placeholder="Enter your name")
        if st.sidebar.button("Start Session"):
            if username:
                st.session_state.user_authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.sidebar.error("Please enter a username")
    else:
        st.sidebar.success(f"👋 Welcome, **{st.session_state.username}**!")
        if st.sidebar.button("Logout"):
            st.session_state.user_authenticated = False
            st.session_state.username = ""
            st.rerun()

def navigation_sidebar():
    """Enhanced navigation sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    
    nav_options = {
        "🏠 Real-Time Dashboard": "Real-Time Dashboard", 
        "🔍 Manual Analysis": "Manual Analysis",
        "📊 Analysis History": "History",
        "📈 Health Trends": "Trends",
        "📚 Bee Health Guide": "Guide",
        "⚙️ Settings": "Settings"
    }
    
    for option, page in nav_options.items():
        if st.sidebar.button(option, use_container_width=True):
            st.session_state.current_page = page
            st.rerun()

def real_time_dashboard():
    """Real-time monitoring dashboard - UPDATED WITH SYNC FUNCTIONALITY"""
    st.markdown('<h1 class="main-header">🐝 Real-Time Hive Monitoring</h1>', unsafe_allow_html=True)
    
    # Sync and refresh controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.session_state.auto_refresh = st.checkbox("🔄 Auto-refresh every 30 seconds", value=True)
    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    with col3:
        if st.button("🔄 Sync Storage"):
            with st.spinner("Syncing storage files to database..."):
                synced_count = sync_storage_to_database()
                if synced_count > 0:
                    st.success(f"✅ Synced {synced_count} files from storage to database!")
                else:
                    st.info("No new files to sync")
            st.rerun()
    with col4:
        st.metric("Last Update", datetime.datetime.now().strftime("%H:%M:%S"))
    
    # Get recent data
    recordings = get_recent_recordings(10)  # Increased limit to show more
    analyses = get_analysis_results(10)
    
    # Current status overview
    st.subheader("📊 Current Hive Status")
    
    if analyses:
        latest_analysis = analyses[0]
        status = latest_analysis.get('health_status', 'Unknown')
        emoji, color_class = get_status_color(status)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f'<div class="{color_class}"><h2>{emoji} {status}</h2></div>', unsafe_allow_html=True)
        
        with col2:
            if 'audio_recordings' in latest_analysis and latest_analysis['audio_recordings']:
                try:
                    recording_time = datetime.datetime.fromisoformat(latest_analysis['audio_recordings']['recording_time']).strftime("%H:%M")
                    st.metric("Last Recording", recording_time)
                except Exception:
                    st.metric("Last Recording", "N/A")
            else:
                st.metric("Last Recording", "N/A")
        
        with col3:
            if 'confidence_score' in latest_analysis:
                try:
                    c = float(latest_analysis['confidence_score'])
                    if 0 <= c <= 1:
                        st.metric("Confidence", f"{int(c * 100)}%")
                    else:
                        st.metric("Confidence", f"{c}")
                except Exception:
                    st.metric("Confidence", "N/A")
            else:
                st.metric("Confidence", "N/A")
    else:
        st.info("No analysis data available yet. Waiting for recordings...")
    
    # Recent recordings from ESP32 - UPDATED TO SHOW STORAGE FILES
    st.subheader("🎵 Recent ESP32 Recordings")
    
    if recordings:
        for recording in recordings[:8]:  # Show more recordings
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
            
            with col1:
                device_name = recording.get('devices', {}).get('name', 'ESP32 Device')
                file_name = recording.get('file_name', recording.get('storage_path', 'Unknown'))
                st.write(f"**{device_name}**")
                st.write(f"📁 {file_name}")
            
            with col2:
                try:
                    recording_time = datetime.datetime.fromisoformat(recording['recording_time']).strftime("%H:%M")
                except Exception:
                    recording_time = "N/A"
                st.write(f"🕒 {recording_time}")
            
            with col3:
                duration = recording.get('duration_seconds', 0)
                st.write(f"⏱️ {duration}s")
            
            with col4:
                # Show source type
                if recording.get('storage_path'):
                    st.write("📍 Storage")
                elif recording.get('audio_data_base64'):
                    st.write("📍 Database")
                else:
                    st.write("📍 Unknown")
            
            with col5:
                # Check if this recording has been analyzed
                analyzed = any('recording_id' in analysis and analysis['recording_id'] == recording['id'] for analysis in analyses)
                if analyzed:
                    st.success("✅ Analyzed")
                else:
                    if st.button("Analyze", key=f"analyze_{recording['id']}"):
                        with st.spinner("Analyzing..."):
                            result = analyze_supabase_recording(recording)
                            if result:
                                save_analysis_result(recording['id'], result)
                                st.success("✅ Analysis saved!")
                                st.rerun()
    
    # Health trends (keep existing code)
    st.subheader("📈 Health Trends")
    
    if analyses:
        trend_data = []
        for analysis in analyses:
            try:
                trend_data.append({
                    'timestamp': datetime.datetime.fromisoformat(analysis['created_at']),
                    'status': analysis['health_status'],
                    'confidence': analysis.get('confidence_score', 0.8)
                })
            except Exception:
                continue
        
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            
            fig = go.Figure()
            
            status_colors = {
                'Healthy': '#4CAF50',
                'Warning': '#FF9800', 
                'Danger': '#F44336'
            }
            
            for status in trend_df['status'].unique():
                status_data = trend_df[trend_df['status'] == status]
                fig.add_trace(go.Scatter(
                    x=status_data['timestamp'],
                    y=[1] * len(status_data),
                    mode='markers',
                    name=status,
                    marker=dict(size=15, color=status_colors.get(status, '#999999')),
                    hovertemplate=f"Status: {status}<br>Time: %{{x}}<extra></extra>"
                ))
            
            fig.update_layout(
                title="Hive Health Status Over Time",
                xaxis_title="Time",
                yaxis=dict(showticklabels=False, title=""),
                height=200,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            status_counts = trend_df['status'].value_counts()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                healthy_count = status_counts.get('Healthy', 0)
                st.metric("🟢 Healthy", healthy_count)
            
            with col2:
                warning_count = status_counts.get('Warning', 0)
                st.metric("🟡 Warning", warning_count)
            
            with col3:
                danger_count = status_counts.get('Danger', 0)
                st.metric("🔴 Danger", danger_count)
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(30)
        st.rerun()

# Keep all other page functions the same (they don't need changes)
def manual_analysis_page():
    """Enhanced hive analysis page"""
    st.header("🔍 Manual Hive Analysis")
    
    analysis_type = st.radio(
        "Select Analysis Type:",
        ["Single Hive Analysis", "Multiple Hive Comparison", "Demo Mode"],
        horizontal=True
    )
    
    if analysis_type == "Single Hive Analysis":
        single_hive_analysis()
    elif analysis_type == "Multiple Hive Comparison":
        multiple_hive_analysis()
    else:
        demo_mode()

def single_hive_analysis():
    """Single hive analysis interface"""
    st.subheader("🎵 Upload Single Hive Recording")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac'],
            help="Upload a recording from your bee hive",
            key="single_upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            st.audio(uploaded_file, format='audio/wav')
            
            with st.expander("🎯 Recording Guidlines : "):
                st.markdown("""
                **Optimal Recording Conditions:**
                - 🕒 **Time:** Morning hours (9-11 AM)
                - 🌤️ **Weather:** Calm, sunny days
                - 📍 **Position:** Close to hive entrance
                - ⏱️ **Duration:** 1-5 minutes
                - 🔇 **Noise:** Minimal background noise
                """)
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if uploaded_file and st.button("🚀 Analyze Hive Health", use_container_width=True):
            with st.spinner("🔬 Analyzing bee sounds patterns..."):
                feature_data, file_names = analyze_uploaded_audio([uploaded_file])
                
                if feature_data:
                    synthetic_data = generate_synthetic_bee_sounds(50)
                    all_features = feature_data + synthetic_data
                    cleaned_features = clean_feature_data(all_features)
                    
                    features_df = pd.DataFrame(cleaned_features)
                    clustering_df = features_df.select_dtypes(include=[np.number])
                    clustering_df = clustering_df.fillna(0)
                    
                    analyzer = BeeClusterAnalyzer()
                    features_scaled = analyzer.prepare_features(clustering_df)
                    
                    try:
                        if isinstance(features_scaled, np.ndarray) and (np.isnan(features_scaled).any() or np.isinf(features_scaled).any()):
                            features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:
                        pass
                    
                    optimal_clusters = max(2, min(3, len(features_scaled) - 1))
                    cluster_labels = analyzer.cluster_analysis(features_scaled, n_clusters=optimal_clusters)
                    cluster_profiles = analyzer.interpret_clusters(features_scaled, cluster_labels)
                    
                    real_sample_cluster = int(cluster_labels[0])
                    status = cluster_profiles.get(real_sample_cluster, {}).get('inferred_status', 'Unknown')
                    
                    emoji, color_class = get_status_color(status)
                    
                    st.markdown(f'<div class="{color_class}"><h2>{emoji} {status}</h2></div>', unsafe_allow_html=True)
                    
                    profile = cluster_profiles.get(real_sample_cluster, {})
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        if 'fundamental_freq_mean' in profile:
                            try:
                                freq = profile['fundamental_freq_mean']['mean']
                                st.metric("🐝 Average Frequency", f"{freq:.1f} Hz")
                            except Exception:
                                pass
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with metrics_col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        if 'harmonic_noise_ratio' in profile:
                            try:
                                hnr = profile['harmonic_noise_ratio']['mean']
                                st.metric("🎵 Harmony Score", f"{hnr:.2f}")
                            except Exception:
                                pass
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with metrics_col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        if 'activity_ratio' in profile:
                            try:
                                activity = profile['activity_ratio']['mean']
                                st.metric("🔊 Activity Level", f"{activity:.2f}")
                            except Exception:
                                pass
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.subheader("📈 Pattern Analysis")
                    
                    try:
                        visualizer = BeeHealthVisualizer()
                        reduced_features = analyzer.reduce_dimensionality(features_scaled)
                        fig = visualizer.plot_cluster_analysis(reduced_features, cluster_labels, cluster_profiles)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Visualization error: {str(e)}")
                    
                    st.markdown("---")
                    st.subheader("📝 Analysis Explanation")
                    
                    explanation = interpret_graph_for_beekeeper(cluster_profiles, real_sample_cluster)
                    st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)
                    
                    st.subheader("📋 Recommended Actions")
                    if 'healthy' in status.lower():
                        st.success("""
                        ✅ **Your hive appears healthy! Continue current practices:**
                        - Regular hive inspections (weekly)
                        - Monitor food stores
                        - Check for pests weekly
                        - Maintain water source nearby
                        """)
                    elif 'queenless' in status.lower():
                        st.warning("""
                        ⚠️ **Possible queen issues detected:**
                        - Check for fresh eggs and larvae
                        - Look for queen cells
                        - Consider requeening if confirmed
                        - Monitor brood pattern closely
                        - Consult experienced beekeeper
                        """)
                    elif 'stress' in status.lower():
                        st.warning("""
                        🔥 **Hive shows stress signs:**
                        - Ensure adequate ventilation
                        - Provide water source nearby
                        - Check for mite infestation
                        - Reduce hive disturbances
                        - Monitor temperature levels
                        """)
                    
                    analysis_data = {
                        'files': file_names,
                        'status': status,
                        'cluster': real_sample_cluster,
                        'metrics': {
                            'Frequency': f"{profile.get('fundamental_freq_mean', {}).get('mean', 0):.1f} Hz",
                            'Harmony': f"{profile.get('harmonic_noise_ratio', {}).get('mean', 0):.2f}",
                            'Activity': f"{profile.get('activity_ratio', {}).get('mean', 0):.2f}"
                        },
                        'summary': f"Single hive analysis - {status}"
                    }
                    add_to_history(analysis_data)
                    
        elif not uploaded_file:
            st.info("""
            👆 **Upload a bee hive recording to get started**
            
            **What happens next:**
            1. Upload your audio file
            2. AI analyzes sound patterns
            3. Get instant health assessment
            4. Receive actionable recommendations
            """)

def multiple_hive_analysis():
    """Multiple hive comparison interface"""
    st.header("📊 Multiple Hive Comparison")
    
    st.subheader("Upload Recordings from Multiple Hives")
    uploaded_files = st.file_uploader(
        "Choose multiple audio files",
        type=['wav', 'mp3', 'm4a', 'flac'],
        accept_multiple_files=True,
        help="Upload recordings from different hives for comparison"
    )
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} files uploaded successfully")
        
        for i, uploaded_file in enumerate(uploaded_files):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Hive {i+1}:** {uploaded_file.name}")
            with col2:
                st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🔍 Analyze All Hives"):
            with st.spinner("Analyzing multiple hives..."):
                feature_data, file_names = analyze_uploaded_audio(uploaded_files)
                
                if len(feature_data) > 0:
                    if len(feature_data) >= 5:
                        all_features = feature_data
                    else:
                        synthetic_data = generate_synthetic_bee_sounds(30)
                        all_features = feature_data + synthetic_data
                    
                    cleaned_features = clean_feature_data(all_features)
                    features_df = pd.DataFrame(cleaned_features)
                    clustering_df = features_df.select_dtypes(include=[np.number])
                    clustering_df = clustering_df.fillna(0)
                    
                    analyzer = BeeClusterAnalyzer()
                    features_scaled = analyzer.prepare_features(clustering_df)
                    
                    try:
                        if isinstance(features_scaled, np.ndarray) and (np.isnan(features_scaled).any() or np.isinf(features_scaled).any()):
                            features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:
                        pass
                    
                    optimal_clusters = max(2, min(4, len(features_scaled) - 1))
                    cluster_labels = analyzer.cluster_analysis(features_scaled, n_clusters=optimal_clusters)
                    cluster_profiles = analyzer.interpret_clusters(features_scaled, cluster_labels)
                    
                    st.subheader("🏆 Hive Health Summary")
                    
                    results_data = []
                    for i, file_name in enumerate(file_names):
                        cluster_id = int(cluster_labels[i])
                        status = cluster_profiles.get(cluster_id, {}).get('inferred_status', 'Unknown')
                        emoji, _ = get_status_color(status)
                        results_data.append({
                            'Hive': file_name,
                            'Status': f"{emoji} {status}",
                            'Cluster': f"Group {cluster_id + 1}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    st.subheader("📈 Health Distribution")
                    status_counts = pd.DataFrame(results_data)['Status'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#4CAF50' if 'Healthy' in str(x) else 
                             '#FF9800' if 'Warning' in str(x) else 
                             '#F44336' for x in status_counts.index]
                    
                    bars = ax.bar(status_counts.index, status_counts.values, color=colors)
                    ax.set_ylabel('Number of Hives')
                    ax.set_title('Hive Health Distribution')
                    plt.xticks(rotation=45, ha='right')
                    
                    for bar, value in zip(bars, status_counts.values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               str(value), ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    
                    st.subheader("🔍 Pattern Clusters")
                    try:
                        visualizer = BeeHealthVisualizer()
                        reduced_features = analyzer.reduce_dimensionality(features_scaled)
                        fig = visualizer.plot_cluster_analysis(reduced_features, cluster_labels, cluster_profiles)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Visualization error: {str(e)}")

def demo_mode():
    """Demo mode with synthetic data"""
    st.header("🎮 Demo Mode")
    
    st.info("""
    Try out HiveMind AI with synthetic data to see how it works!
    This mode generates simulated bee hive recordings to demonstrate the analysis capabilities.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_samples = st.slider("Number of samples to generate", 10, 200, 100)
        num_clusters = st.slider("Number of clusters to find", 2, 6, 3)
    
    with col2:
        st.subheader("Simulated Conditions")
        st.markdown("""
        - 🟢 Healthy & Productive
        - 🟡 Stressed & Agitated  
        - 🔴 Queenless & Irregular
        """)
    
    if st.button("🚀 Run Demo Analysis"):
        with st.spinner("Generating and analyzing synthetic bee data..."):
            feature_data = generate_synthetic_bee_sounds(num_samples)
            features_df = pd.DataFrame(feature_data)
            clustering_df = features_df.select_dtypes(include=[np.number])
            
            analyzer = BeeClusterAnalyzer()
            features_scaled = analyzer.prepare_features(clustering_df)
            cluster_labels = analyzer.cluster_analysis(features_scaled, n_clusters=num_clusters)
            cluster_profiles = analyzer.interpret_clusters(features_scaled, cluster_labels)
            
            st.subheader("📊 Demo Results")
            
            for cluster_id, profile in cluster_profiles.items():
                status = profile.get('inferred_status', 'Unknown')
                cluster_count = np.sum(cluster_labels == cluster_id)
                percentage = (cluster_count / len(cluster_labels)) * 100
                
                emoji, color_class = get_status_color(status)
                
                with st.expander(f"{emoji} Cluster {cluster_id}: {status} ({cluster_count} samples, {percentage:.1f}%)", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'fundamental_freq_mean' in profile:
                            freq = profile['fundamental_freq_mean']['mean']
                            st.metric("Avg Frequency", f"{freq:.1f} Hz")
                    
                    with col2:
                        if 'harmonic_noise_ratio' in profile:
                            hnr = profile['harmonic_noise_ratio']['mean']
                            st.metric("Harmony Score", f"{hnr:.2f}")
                    
                    with col3:
                        if 'activity_ratio' in profile:
                            activity = profile['activity_ratio']['mean']
                            st.metric("Activity Level", f"{activity:.2f}")
            
            st.subheader("📈 Pattern Visualization")
            try:
                visualizer = BeeHealthVisualizer()
                reduced_features = analyzer.reduce_dimensionality(features_scaled)
                fig = visualizer.plot_cluster_analysis(reduced_features, cluster_labels, cluster_profiles)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
            
            st.success("🎉 Demo completed successfully! This shows how HiveMind AI would analyze real bee hive recordings.")

def history_page():
    """Analysis history page"""
    st.header("📊 Analysis History")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history yet. Start by analyzing some hive recordings!")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        filter_status = st.selectbox("Filter by status", ["All", "Healthy", "Warning", "Danger"])
    with col2:
        search_term = st.text_input("Search files...")
    
    for i, analysis in enumerate(st.session_state.analysis_history):
        if filter_status != "All":
            status_lower = analysis['status'].lower()
            filter_lower = filter_status.lower()
            if filter_lower not in status_lower:
                continue
        
        if search_term and search_term.lower() not in str(analysis['files']).lower():
            continue
        
        with st.expander(f"{analysis['timestamp']} - {analysis['status']}", expanded=i==0):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Files:** {', '.join(analysis['files'])}")
                st.write(f"**Status:** {analysis['status']}")
                st.write(f"**Pattern Group:** {analysis['cluster'] + 1}")
            
            with col2:
                if analysis['metrics']:
                    st.write("**Key Metrics:**")
                    for metric, value in analysis['metrics'].items():
                        st.write(f"- {metric}: {value}")

def trends_page():
    """Historical trends analysis"""
    st.header("📈 Historical Health Trends")
    
    analyses = get_analysis_results(50)
    
    if not analyses:
        st.info("No analysis data available for trends.")
        return
    
    trend_data = []
    for analysis in analyses:
        try:
            trend_data.append({
                'date': datetime.datetime.fromisoformat(analysis['created_at']).date(),
                'timestamp': datetime.datetime.fromisoformat(analysis['created_at']),
                'status': analysis['health_status'],
                'confidence': analysis.get('confidence_score', 0.8),
                'frequency': analysis.get('fundamental_freq_mean', 0),
                'harmony': analysis.get('harmonic_noise_ratio', 0),
                'activity': analysis.get('activity_ratio', 0)
            })
        except Exception:
            continue
    
    df = pd.DataFrame(trend_data)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=df['date'].min())
    with col2:
        end_date = st.date_input("End Date", value=df['date'].max())
    
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    if filtered_df.empty:
        st.warning("No data in selected date range.")
        return
    
    st.subheader("Health Status Timeline")
    
    fig = px.scatter(filtered_df, x='timestamp', y='status', 
                    color='status', color_discrete_map={
                        'Healthy': '#4CAF50',
                        'Warning': '#FF9800',
                        'Danger': '#F44336'
                    })
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Health Status",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Key Metrics Over Time")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        fig1 = px.line(filtered_df, x='timestamp', y='frequency', title='Frequency Trend')
        st.plotly_chart(fig1, use_container_width=True)
    
    with metric_col2:
        fig2 = px.line(filtered_df, x='timestamp', y='harmony', title='Harmony Score Trend')
        st.plotly_chart(fig2, use_container_width=True)
    
    with metric_col3:
        fig3 = px.line(filtered_df, x='timestamp', y='activity', title='Activity Level Trend')
        st.plotly_chart(fig3, use_container_width=True)

def guide_page():
    """Bee health guide page"""
    st.header("📚 Bee Health Guide")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🐝 Bee Sounds", "🔍 Health Indicators", "📱 Using the App", "❓ FAQ"])
    
    with tab1:
        st.subheader("Understanding Bee Sounds")
        st.markdown("""
        **Normal Bee Sounds:**
        - **Healthy Buzz:** Consistent 200-250 Hz humming
        - **Content Bees:** Steady, harmonious patterns
        - **Normal Activity:** Moderate sound levels with rhythm
        
        **Concerning Sounds:**
        - **Queenless:** Irregular, chaotic patterns
        - **Stressed:** High-pitched, agitated buzzing  
        - **Swarming:** Intense, synchronized activity
        - **Sick:** Weak, irregular buzzing
        """)
    
    with tab2:
        st.subheader("Health Indicators")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **🟢 Healthy Signs:**
            - Consistent buzzing
            - Normal activity levels
            - Good harmony score
            - Steady frequency
            """)
        
        with col2:
            st.error("""
            **🔴 Warning Signs:**
            - Irregular patterns
            - Extreme frequencies
            - Low harmony
            - Unusual activity
            """)
    
    with tab3:
        st.subheader("Using the App")
        st.markdown("""
        **Step-by-Step Guide:**
        1. **Record:** Use ESP32 near hive entrance
        2. **Upload:** ESP32 sends to Supabase automatically
        3. **Sync:** Click 'Sync Storage' in Real-Time Dashboard
        4. **Analyze:** AI processes sound patterns
        5. **Review:** Get health status & recommendations
        
        **Best Practices:**
        - Record at same time each day
        - Note weather conditions
        - Track changes over time
        - Compare multiple hives
        """)
    
    with tab4:
        st.subheader("Frequently Asked Questions")
        
        with st.expander("How accurate is the analysis?"):
            st.write("The AI is trained on bee sound patterns and provides reliable indicators. However, always verify with visual hive inspection.")
        
        with st.expander("What if I get different results each time?"):
            st.write("Bee sounds can vary throughout the day. Try recording at consistent times and note environmental factors.")
        
        with st.expander("Can I use this for commercial beekeeping?"):
            st.write("Yes! Many commercial beekeepers use similar technology to monitor hive health at scale.")

def settings_page():
    """Settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("User Preferences")
    st.text_input("Your Name", value=st.session_state.username, disabled=True)
    st.selectbox("Temperature Units", ["Celsius", "Fahrenheit"])
    st.selectbox("Language", ["English", "Spanish", "French", "German"])
    
    st.subheader("Data Management")
    if st.button("Clear Analysis History"):
        st.session_state.analysis_history = []
        st.success("History cleared!")
    
    st.subheader("About HiveMind AI")
    st.markdown("""
    **Version:** 2.0.0  
    **Developed by:** Bee Health Technologies  
    **Contact:** support@hivemind-ai.com  
    **Mission:** Helping beekeepers protect their colonies through AI
    """)

def main():
    # User authentication
    login_section()
    
    # Navigation
    navigation_sidebar()
    
    # Page routing
    if st.session_state.current_page == "Real-Time Dashboard":
        real_time_dashboard()
    elif st.session_state.current_page == "Manual Analysis":
        manual_analysis_page()
    elif st.session_state.current_page == "History":
        history_page()
    elif st.session_state.current_page == "Trends":
        trends_page()
    elif st.session_state.current_page == "Guide":
        guide_page()
    elif st.session_state.current_page == "Settings":
        settings_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🐝 <b>HiveMind AI</b> - Professional Bee Health Monitoring System</p>
        <p><small>Helping beekeepers make informed decisions through AI-powered sound analysis</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
