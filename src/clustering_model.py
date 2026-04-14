import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class BeeClusterAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.feature_names = None
        
    def prepare_features(self, feature_data):
        """Convert feature dictionary to array and scale"""
        try:
            if isinstance(feature_data, list):
                # List of dictionaries
                df = pd.DataFrame(feature_data)
            else:
                df = feature_data
                
            self.feature_names = df.columns.tolist()
            features_scaled = self.scaler.fit_transform(df)
            return features_scaled
        except Exception as e:
            print(f"Error preparing features: {e}")
            # Return identity matrix as fallback
            if isinstance(feature_data, list):
                size = len(feature_data)
            else:
                size = len(feature_data)
            return np.eye(size)
    
    def find_optimal_clusters(self, features, max_clusters=8):
        """Find optimal number of clusters using silhouette score"""
        try:
            if len(features) < 3:
                return 2, [], []
                
            silhouette_scores = []
            
            for n_clusters in range(2, min(max_clusters + 1, len(features))):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(features, cluster_labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(0)
            
            if silhouette_scores:
                optimal_clusters = np.argmax(silhouette_scores) + 2
            else:
                optimal_clusters = 2
                
            return optimal_clusters, [], silhouette_scores
        except Exception as e:
            print(f"Error finding optimal clusters: {e}")
            return 3, [], []
    
    def cluster_analysis(self, features, n_clusters=None):
        """Perform clustering analysis"""
        try:
            if n_clusters is None:
                n_clusters, _, _ = self.find_optimal_clusters(features)
            
            n_clusters = max(2, min(n_clusters, len(features) - 1))
            print(f"Using {n_clusters} clusters for analysis")
            
            # Use KMeans for clustering
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.cluster_model.fit_predict(features)
            
            return cluster_labels
        except Exception as e:
            print(f"Error in cluster analysis: {e}")
            # Return random clusters as fallback
            return np.random.randint(0, 2, len(features))
    
    def reduce_dimensionality(self, features, method='pca'):
        """Reduce features to 2D for visualization"""
        try:
            if method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
            elif method == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
            else:  # Try UMAP if available
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42)
                except ImportError:
                    print("UMAP not available, using PCA instead")
                    reducer = PCA(n_components=2, random_state=42)
                    
            reduced_features = reducer.fit_transform(features)
            return reduced_features
        except Exception as e:
            print(f"Dimensionality reduction failed: {e}")
            print("Using simple PCA as fallback...")
            reducer = PCA(n_components=2, random_state=42)
            return reducer.fit_transform(features)
    
    def interpret_clusters(self, features, cluster_labels):
        """Interpret what each cluster might represent based on feature patterns"""
        try:
            df = pd.DataFrame(features, columns=self.feature_names)
            df['cluster'] = cluster_labels
            
            cluster_profiles = {}
            
            for cluster_id in np.unique(cluster_labels):
                cluster_data = df[df['cluster'] == cluster_id]
                profile = {}
                
                # Key features for bee health interpretation
                key_features = [
                    'fundamental_freq_mean', 'fundamental_freq_std',
                    'harmonic_noise_ratio', 'activity_ratio',
                    'spectral_centroid_mean', 'rms_energy',
                    'zero_crossing_rate'
                ]
                
                available_features = [f for f in key_features if f in cluster_data.columns]
                
                for feature in available_features:
                    profile[feature] = {
                        'mean': float(cluster_data[feature].mean()),
                        'std': float(cluster_data[feature].std())
                    }
                
                # Make health inferences based on acoustic patterns
                health_status = self._infer_health_status(profile)
                profile['inferred_status'] = health_status
                
                cluster_profiles[int(cluster_id)] = profile
            
            return cluster_profiles
        except Exception as e:
            print(f"Error interpreting clusters: {e}")
            return {0: {'inferred_status': 'Analysis Failed'}}
    
    def _infer_health_status(self, profile):
        """Infer potential bee health status from acoustic patterns"""
        try:
            freq_mean = profile.get('fundamental_freq_mean', {}).get('mean', 220)
            freq_std = profile.get('fundamental_freq_std', {}).get('mean', 0.2)
            hnr = profile.get('harmonic_noise_ratio', {}).get('mean', 1.5)
            activity = profile.get('activity_ratio', {}).get('mean', 0.5)
            
            # Rule-based inference
            if hnr > 2.0 and 0.1 < freq_std < 0.3:
                return "Healthy - Calm & Productive"
            elif freq_std > 0.5 and activity > 0.8:
                return "Stressed - Agitated Activity"
            elif hnr < 1.0 and activity < 0.3:
                return "Queenless - Irregular Pattern"
            elif freq_mean > 280:
                return "Temperature Stress - High Frequency"
            elif activity > 0.9 and freq_std < 0.1:
                return "Swarming Risk - Synchronized Activity"
            elif hnr < 0.5 and freq_std > 0.7:
                return "Parasite Attack - Disrupted Harmony"
            else:
                return "Needs Investigation - Unique Pattern"
                
        except Exception:
            return "Unknown - Insufficient Data"
